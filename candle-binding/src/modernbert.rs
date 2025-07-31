// ModernBERT binding for classification tasks
// Based on ModernBERT implementation in candle-transformers

use std::ffi::{c_char, CStr};
use std::sync::Arc;
use std::sync::Mutex;
use std::path::Path;

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::modernbert::{ModernBert, Config, ClassifierConfig, ClassifierPooling};
use candle_nn::Module;
use candle_core::{D, IndexOp};
use candle_nn::ops;
use std::collections::HashMap;
use tokenizers::{Tokenizer, PaddingParams, PaddingStrategy};
use serde_json;

// ================================================================================================
// FIXED MODERNBERT IMPLEMENTATION
// ================================================================================================
// This implementation fixes the bugs in candle-transformers ModernBERT:
// 1. Proper token ID to embedding conversion
// 2. Correct pooling logic (CLS vs MEAN)
// 3. Proper error handling and validation

/// Fixed ModernBERT classifier that handles embeddings correctly
#[derive(Clone)]
pub struct FixedModernBertClassifier {
    classifier: candle_nn::Linear,
}

impl FixedModernBertClassifier {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let num_classes = config
            .classifier_config
            .as_ref()
            .map(|cc| cc.id2label.len())
            .unwrap_or(2);
            
        let classifier = candle_nn::Linear::new(
            vb.get((num_classes, config.hidden_size), "classifier.weight")?,
            Some(vb.get((num_classes,), "classifier.bias")?),
        );
        
        Ok(Self { classifier })
    }
}

impl Module for FixedModernBertClassifier {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let logits = xs.apply(&self.classifier)?;
        // Apply softmax to get probabilities
        ops::softmax(&logits, D::Minus1)
    }
}

/// Fixed ModernBERT head (dense layer + layer norm)
#[derive(Clone)]
pub struct FixedModernBertHead {
    dense: candle_nn::Linear,
    layer_norm: candle_nn::LayerNorm,
}

impl FixedModernBertHead {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = candle_nn::Linear::new(
            vb.get((config.hidden_size, config.hidden_size), "dense.weight")?,
            None,
        );
        
        // Load layer norm - it's called "norm" not "layer_norm" in this model!
        // And no bias based on actual model inspection
        let layer_norm = candle_nn::LayerNorm::new(
            vb.get((config.hidden_size,), "norm.weight")?,
            // Create a zero bias tensor since LayerNorm::new requires it but the model doesn't have one
            Tensor::zeros((config.hidden_size,), DType::F32, &vb.device())?,
            1e-12,
        );
        
        Ok(Self { dense, layer_norm })
    }
}

impl Module for FixedModernBertHead {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = xs.apply(&self.dense)?;
        // Apply GELU activation
        let xs = xs.gelu()?;
        xs.apply(&self.layer_norm)
    }
}

/// Fixed ModernBERT sequence classification model that properly handles embeddings
#[derive(Clone)]
pub struct FixedModernBertForSequenceClassification {
    model: ModernBert, // Use the base model (this should work)
    head: Option<FixedModernBertHead>, // Head might not exist in some ModernBERT models
    classifier: FixedModernBertClassifier,
    classifier_pooling: ClassifierPooling,
}

impl FixedModernBertForSequenceClassification {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let model = ModernBert::load(vb.clone(), config)?;
        
        // Try to load head - it might not exist in all ModernBERT models  
        let head = match vb.get((config.hidden_size, config.hidden_size), "head.dense.weight") {
            Ok(_) => {
                let head_vb = vb.pp("head");
                Some(FixedModernBertHead::load(head_vb, config)?)
            }
            Err(_) => None
        };
        
        let classifier = FixedModernBertClassifier::load(vb.clone(), config)?;
        
        let classifier_pooling = config
            .classifier_config
            .as_ref()
            .map(|cc| cc.classifier_pooling)
            .unwrap_or(ClassifierPooling::CLS);
            
        Ok(Self {
            model,
            head,
            classifier,
            classifier_pooling,
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // Get embeddings from the base model
        let output = self.model.forward(xs, mask).map_err(|e| {
            let error_str = format!("{}", e);
                E::msg(format!("Base model failed: {}", error_str))
        })?;
        
        // Apply correct pooling logic
        let pooled = match self.classifier_pooling {
            ClassifierPooling::CLS => {
                output.i((.., 0, ..))?
            }
            ClassifierPooling::MEAN => {
                let mask_expanded = mask.unsqueeze(D::Minus1)?.to_dtype(DType::F32)?;
                let masked_output = output.broadcast_mul(&mask_expanded)?;
                let sum_output = masked_output.sum(1)?;
                let mask_sum = mask.sum_keepdim(1)?.to_dtype(DType::F32)?;
                sum_output.broadcast_div(&mask_sum)?
            }
        };
        
        // Apply head (dense + layer norm) if it exists
        let classifier_input = match &self.head {
            Some(head) => head.forward(&pooled).map_err(E::msg)?,
            None => pooled,
        };
        
        // Apply classifier (linear + softmax)
        let probabilities = self.classifier.forward(&classifier_input).map_err(E::msg)?;
        
        Ok(probabilities)
    }
}

// Structure to hold ModernBERT model and tokenizer for text classification
pub struct ModernBertClassifier {
    model: FixedModernBertForSequenceClassification,
    tokenizer: Tokenizer,
    device: Device,
    pad_token_id: u32,
}

lazy_static::lazy_static! {
    static ref MODERNBERT_CLASSIFIER: Arc<Mutex<Option<ModernBertClassifier>>> = Arc::new(Mutex::new(None));
    static ref MODERNBERT_PII_CLASSIFIER: Arc<Mutex<Option<ModernBertClassifier>>> = Arc::new(Mutex::new(None));
    static ref MODERNBERT_JAILBREAK_CLASSIFIER: Arc<Mutex<Option<ModernBertClassifier>>> = Arc::new(Mutex::new(None));
}

// Structure to hold classification result
#[repr(C)]
pub struct ModernBertClassificationResult {
    pub class: i32,
    pub confidence: f32,
}

impl ModernBertClassifier {
    pub fn new(model_id: &str, use_cpu: bool) -> Result<Self> {
        Self::new_internal(model_id, use_cpu)
    }

    /// Internal implementation using the fixed ModernBERT
    fn new_internal(model_id: &str, use_cpu: bool) -> Result<Self> {
        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };

        // Check if this is a SentenceTransformer ModernBERT model
        let _is_sentence_transformer = Path::new(model_id).join("modules.json").exists();

        let (config_filename, tokenizer_filename, weights_filename, use_pth) = if Path::new(model_id).exists() {
            // Local model path
            let config_path = Path::new(model_id).join("config.json");
            let tokenizer_path = Path::new(model_id).join("tokenizer.json");
            
            // Check for safetensors first, fall back to PyTorch
            let weights_path = if Path::new(model_id).join("model.safetensors").exists() {
                (Path::new(model_id).join("model.safetensors").to_string_lossy().to_string(), false)
            } else if Path::new(model_id).join("pytorch_model.bin").exists() {
                (Path::new(model_id).join("pytorch_model.bin").to_string_lossy().to_string(), true)
            } else {
                return Err(E::msg(format!("No model weights found in {}", model_id)));
            };
            
            (
                config_path.to_string_lossy().to_string(),
                tokenizer_path.to_string_lossy().to_string(),
                weights_path.0,
                weights_path.1
            )
        } else {
            return Err(E::msg("HuggingFace Hub loading for ModernBERT not yet implemented"));
        };

        let config_str = std::fs::read_to_string(&config_filename)?;
        let config: Config = serde_json::from_str(&config_str)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let vb = if use_pth {
            VarBuilder::from_pth(&weights_filename, DType::F32, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)? }
        };
        
        // Check if we have id2label and label2id mappings either in classifier_config or at the top level
        let mut config = config;
        
        // Check if classifier_config exists and has mappings
        let has_classifier_config = config.classifier_config
            .as_ref()
            .map(|cc| !cc.id2label.is_empty())
            .unwrap_or(false);
        
        // If no classifier_config or it's empty, check for top-level id2label/label2id
        if !has_classifier_config {
            // Try to access top-level id2label and label2id fields
            
            let config_str = std::fs::read_to_string(config_filename)?;
            let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
            
            if let (Some(id2label), Some(label2id)) = (
                config_json.get("id2label").and_then(|v| v.as_object()),
                config_json.get("label2id").and_then(|v| v.as_object())
            ) {
                // Convert JSON objects to HashMap<String, String>
                let id2label_map: HashMap<String, String> = id2label
                    .iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("UNKNOWN").to_string()))
                    .collect();
                    
                let label2id_map: HashMap<String, String> = label2id
                    .iter()
                    .map(|(k, v)| (k.clone(), v.as_i64().unwrap_or(0).to_string()))
                    .collect();
                
                // Extract classifier_pooling from top-level config
                let classifier_pooling = config_json.get("classifier_pooling")
                    .and_then(|v| v.as_str())
                    .map(|s| match s {
                        "cls" => ClassifierPooling::CLS,
                        "mean" => ClassifierPooling::MEAN,
                        _ => ClassifierPooling::CLS, // Default to CLS
                    })
                    .unwrap_or(ClassifierPooling::CLS);
                
                let classifier_config = ClassifierConfig {
                    id2label: id2label_map,
                    label2id: label2id_map,
                    classifier_pooling,
                };
                
                config.classifier_config = Some(classifier_config);
            } else {
                return Err(E::msg("No id2label/label2id mappings found in config - required for classification"));
            }
        }
        
        // Load the fixed ModernBERT model directly
        let model = FixedModernBertForSequenceClassification::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            pad_token_id: config.pad_token_id,
        })
    }

    pub fn classify_text(&self, text: &str) -> Result<(usize, f32)> {
        // Set up tokenizer
        let mut tokenizer = self.tokenizer.clone();
        
        // Set up padding - use config's pad_token_id and no truncation
        tokenizer
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                pad_id: self.pad_token_id,
                ..Default::default()
            }))
            .with_truncation(None)
            .map_err(E::msg)?;

        // Tokenize input text
        let tokens = tokenizer.encode_batch(vec![text], true).map_err(E::msg)?;
        
        // Create tensors - convert to u32 for ModernBERT
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens: Vec<u32> = tokens.get_ids().iter().map(|&x| x as u32).collect();
                Tensor::new(tokens.as_slice(), &self.device)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        let attention_mask = tokens
            .iter()
            .map(|tokens| {
                let tokens: Vec<u32> = tokens.get_attention_mask().iter().map(|&x| x as u32).collect();
                Tensor::new(tokens.as_slice(), &self.device)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        let input_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        
        // Input validation
        if input_ids.dims().len() != 2 {
            return Err(E::msg(format!("Expected input_ids to have 2 dimensions [batch_size, seq_len], got {:?}", input_ids.dims())));
        }
        if attention_mask.dims().len() != 2 {
            return Err(E::msg(format!("Expected attention_mask to have 2 dimensions [batch_size, seq_len], got {:?}", attention_mask.dims())));
        }
        if input_ids.dims()[0] != attention_mask.dims()[0] || input_ids.dims()[1] != attention_mask.dims()[1] {
            return Err(E::msg(format!("input_ids and attention_mask must have same shape, got {:?} vs {:?}", input_ids.dims(), attention_mask.dims())));
        }

        // Run through ModernBERT model
        let probabilities = self.model.forward(&input_ids, &attention_mask)?;
        
        // Remove batch dimension if present
        let probabilities = if probabilities.dims().len() > 1 {
            probabilities.squeeze(0)?
        } else {
            probabilities
        };
        
        // Convert to vector and find the class with highest probability
        let probabilities_vec = probabilities.to_vec1::<f32>()?;
        
        // Get the predicted class with highest probability
        let (predicted_idx, &max_prob) = probabilities_vec.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        
        Ok((predicted_idx, max_prob))
    }
}

// Initialize the ModernBERT classifier model (called from Go)
#[no_mangle]
pub extern "C" fn init_modernbert_classifier(model_id: *const c_char, use_cpu: bool) -> bool {
    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    match ModernBertClassifier::new(model_id, use_cpu) {
        Ok(classifier) => {
            let mut bert_opt = MODERNBERT_CLASSIFIER.lock().unwrap();
            *bert_opt = Some(classifier);
            true
        }
        Err(e) => {
            eprintln!("Failed to initialize ModernBERT classifier: {}", e);
            false
        }
    }
}

// Initialize the ModernBERT PII classifier model (called from Go)
#[no_mangle]
pub extern "C" fn init_modernbert_pii_classifier(model_id: *const c_char, use_cpu: bool) -> bool {
    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    match ModernBertClassifier::new(model_id, use_cpu) {
        Ok(classifier) => {
            let mut bert_opt = MODERNBERT_PII_CLASSIFIER.lock().unwrap();
            *bert_opt = Some(classifier);
            true
        }
        Err(e) => {
            eprintln!("Failed to initialize ModernBERT PII classifier: {}", e);
            false
        }
    }
}

// Initialize the ModernBERT jailbreak classifier model (called from Go)
#[no_mangle]
pub extern "C" fn init_modernbert_jailbreak_classifier(model_id: *const c_char, use_cpu: bool) -> bool {
    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    match ModernBertClassifier::new(model_id, use_cpu) {
        Ok(classifier) => {
            let mut bert_opt = MODERNBERT_JAILBREAK_CLASSIFIER.lock().unwrap();
            *bert_opt = Some(classifier);
            true
        }
        Err(e) => {
            eprintln!("Failed to initialize ModernBERT jailbreak classifier: {}", e);
            false
        }
    }
}

// Classify text using ModernBERT (called from Go)
#[no_mangle]
pub extern "C" fn classify_modernbert_text(text: *const c_char) -> ModernBertClassificationResult {
    let default_result = ModernBertClassificationResult {
        class: -1,
        confidence: 0.0,
    };

    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    let bert_opt = MODERNBERT_CLASSIFIER.lock().unwrap();
    match &*bert_opt {
        Some(classifier) => match classifier.classify_text(text) {
            Ok((class_idx, confidence)) => ModernBertClassificationResult {
                class: class_idx as i32,
                confidence,
            },
            Err(e) => {
                eprintln!("Error classifying text with ModernBERT: {}", e);
                default_result
            }
        },
        None => {
            eprintln!("ModernBERT classifier not initialized");
            default_result
        }
    }
}

// Classify text for PII using ModernBERT (called from Go)
#[no_mangle]
pub extern "C" fn classify_modernbert_pii_text(text: *const c_char) -> ModernBertClassificationResult {
    let default_result = ModernBertClassificationResult {
        class: -1,
        confidence: 0.0,
    };

    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    let bert_opt = MODERNBERT_PII_CLASSIFIER.lock().unwrap();
    match &*bert_opt {
        Some(classifier) => match classifier.classify_text(text) {
            Ok((class_idx, confidence)) => ModernBertClassificationResult {
                class: class_idx as i32,
                confidence,
            },
            Err(e) => {
                eprintln!("Error classifying PII text with ModernBERT: {}", e);
                default_result
            }
        },
        None => {
            eprintln!("ModernBERT PII classifier not initialized");
            default_result
        }
    }
}

// Classify text for jailbreak detection using ModernBERT (called from Go)
#[no_mangle]
pub extern "C" fn classify_modernbert_jailbreak_text(text: *const c_char) -> ModernBertClassificationResult {
    let default_result = ModernBertClassificationResult {
        class: -1,
        confidence: 0.0,
    };

    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    let bert_opt = MODERNBERT_JAILBREAK_CLASSIFIER.lock().unwrap();
    match &*bert_opt {
        Some(classifier) => match classifier.classify_text(text) {
            Ok((class_idx, confidence)) => ModernBertClassificationResult {
                class: class_idx as i32,
                confidence,
            },
            Err(e) => {
                eprintln!("Error classifying jailbreak text with ModernBERT: {}", e);
                default_result
            }
        },
        None => {
            eprintln!("ModernBERT jailbreak classifier not initialized");
            default_result
        }
    }
}
