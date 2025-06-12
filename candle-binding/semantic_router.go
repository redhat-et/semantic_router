package candle_binding

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

/*
#cgo LDFLAGS: -L${SRCDIR}/target/release -lcandle_semantic_router -ldl -lm
#include <stdlib.h>
#include <stdbool.h>

extern bool init_similarity_model(const char* model_id, bool use_cpu);

extern float calculate_similarity(const char* text1, const char* text2, int max_length);

extern bool init_classifier(const char* model_id, int num_classes, bool use_cpu);

extern bool init_pii_classifier(const char* model_id, int num_classes, bool use_cpu);

extern bool init_jailbreak_classifier(const char* model_id, int num_classes, bool use_cpu);

// Similarity result structure
typedef struct {
    int index;
    float score;
} SimilarityResult;

// Embedding result structure
typedef struct {
    float* data;
    int length;
    bool error;
} EmbeddingResult;

// Tokenization result structure
typedef struct {
    int* token_ids;
    int token_count;
    char** tokens;
    bool error;
} TokenizationResult;

// Classification result structure
typedef struct {
    int class;
    float confidence;
} ClassificationResult;

extern SimilarityResult find_most_similar(const char* query, const char** candidates, int num_candidates, int max_length);
extern EmbeddingResult get_text_embedding(const char* text, int max_length);
extern TokenizationResult tokenize_text(const char* text, int max_length);
extern void free_cstring(char* s);
extern void free_embedding(float* data, int length);
extern void free_tokenization_result(TokenizationResult result);
extern ClassificationResult classify_text(const char* text);
extern ClassificationResult classify_pii_text(const char* text);
extern ClassificationResult classify_jailbreak_text(const char* text);
*/
import "C"

var (
	initOnce                    sync.Once
	initErr                     error
	modelInitialized            bool
	classifierInitOnce          sync.Once
	classifierInitErr           error
	piiClassifierInitOnce       sync.Once
	piiClassifierInitErr        error
	jailbreakClassifierInitOnce sync.Once
	jailbreakClassifierInitErr  error
)

// TokenizeResult represents the result of tokenization
type TokenizeResult struct {
	TokenIDs []int32  // Token IDs
	Tokens   []string // String representation of tokens
}

// SimResult represents the result of a similarity search
type SimResult struct {
	Index int     // Index of the most similar text
	Score float32 // Similarity score
}

// ClassResult represents the result of a text classification
type ClassResult struct {
	Class      int     // Class index
	Confidence float32 // Confidence score
}

// InitModel initializes the BERT model with the specified model ID
func InitModel(modelID string, useCPU bool) error {
	var err error
	initOnce.Do(func() {
		if modelID == "" {
			modelID = "sentence-transformers/all-MiniLM-L6-v2"
		}

		fmt.Println("Initializing BERT similarity model:", modelID)

		// Initialize BERT directly using CGO
		cModelID := C.CString(modelID)
		defer C.free(unsafe.Pointer(cModelID))

		success := C.init_similarity_model(cModelID, C.bool(useCPU))
		if !bool(success) {
			err = fmt.Errorf("failed to initialize BERT similarity model")
			return
		}

		modelInitialized = true
	})

	// Reset the once so we can try again with a different model ID if needed
	if err != nil {
		initOnce = sync.Once{}
		modelInitialized = false
	}

	return err
}

// TokenizeText tokenizes the given text into tokens and their IDs with maxLength parameter
func TokenizeText(text string, maxLength int) (TokenizeResult, error) {
	if !modelInitialized {
		return TokenizeResult{}, fmt.Errorf("BERT model not initialized")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	// Pass maxLength parameter to C function to ensure consistent tokenization with Python
	result := C.tokenize_text(cText, C.int(maxLength))

	// Make sure we free the memory allocated by Rust when we're done
	defer C.free_tokenization_result(result)

	if bool(result.error) {
		return TokenizeResult{}, fmt.Errorf("failed to tokenize text")
	}

	// Convert C array of token IDs to Go slice
	tokenCount := int(result.token_count)
	tokenIDs := make([]int32, tokenCount)

	if tokenCount > 0 && result.token_ids != nil {
		// Create a slice that refers to the C array
		cTokenIDs := (*[1 << 30]C.int)(unsafe.Pointer(result.token_ids))[:tokenCount:tokenCount]

		// Copy values
		for i := 0; i < tokenCount; i++ {
			tokenIDs[i] = int32(cTokenIDs[i])
		}
	}

	// Convert C array of token strings to Go slice
	tokens := make([]string, tokenCount)

	if tokenCount > 0 && result.tokens != nil {
		// Create a slice that refers to the C array of char pointers
		cTokens := (*[1 << 30]*C.char)(unsafe.Pointer(result.tokens))[:tokenCount:tokenCount]

		// Convert each C string to Go string
		for i := 0; i < tokenCount; i++ {
			tokens[i] = C.GoString(cTokens[i])
		}
	}

	tokResult := TokenizeResult{
		TokenIDs: tokenIDs,
		Tokens:   tokens,
	}

	return tokResult, nil
}

// TokenizeTextDefault tokenizes text with default max length (512)
func TokenizeTextDefault(text string) (TokenizeResult, error) {
	return TokenizeText(text, 512)
}

// GetEmbedding gets the embedding vector for a text
func GetEmbedding(text string, maxLength int) ([]float32, error) {
	if !modelInitialized {
		return nil, fmt.Errorf("BERT model not initialized")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.get_text_embedding(cText, C.int(maxLength))

	if bool(result.error) {
		return nil, fmt.Errorf("failed to generate embedding")
	}

	// Convert the C array to a Go slice
	length := int(result.length)
	embedding := make([]float32, length)

	if length > 0 {
		// Create a slice that refers to the C array
		cFloats := (*[1 << 30]C.float)(unsafe.Pointer(result.data))[:length:length]

		// Copy and convert each value
		for i := 0; i < length; i++ {
			embedding[i] = float32(cFloats[i])
		}

		// Free the memory allocated in Rust
		C.free_embedding(result.data, result.length)
	}

	return embedding, nil
}

// GetEmbeddingDefault gets the embedding vector for a text with default max length (512)
func GetEmbeddingDefault(text string) ([]float32, error) {
	return GetEmbedding(text, 512)
}

// CalculateSimilarity calculates the similarity between two texts with maxLength parameter
func CalculateSimilarity(text1, text2 string, maxLength int) float32 {
	if !modelInitialized {
		fmt.Println("BERT model not initialized")
		return -1.0
	}

	cText1 := C.CString(text1)
	defer C.free(unsafe.Pointer(cText1))

	cText2 := C.CString(text2)
	defer C.free(unsafe.Pointer(cText2))

	result := C.calculate_similarity(cText1, cText2, C.int(maxLength))
	return float32(result)
}

// CalculateSimilarityDefault calculates the similarity between two texts with default max length (512)
func CalculateSimilarityDefault(text1, text2 string) float32 {
	return CalculateSimilarity(text1, text2, 512)
}

// FindMostSimilar finds the most similar text from a list of candidates with maxLength parameter
func FindMostSimilar(query string, candidates []string, maxLength int) SimResult {
	if !modelInitialized {
		fmt.Println("BERT model not initialized")
		return SimResult{Index: -1, Score: -1.0}
	}

	if len(candidates) == 0 {
		return SimResult{Index: -1, Score: -1.0}
	}

	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))

	// Convert the candidates to C strings
	cCandidates := make([]*C.char, len(candidates))
	for i, candidate := range candidates {
		cCandidates[i] = C.CString(candidate)
		defer C.free(unsafe.Pointer(cCandidates[i]))
	}

	// Create a C array of C strings
	cCandidatesPtr := (**C.char)(unsafe.Pointer(&cCandidates[0]))

	result := C.find_most_similar(cQuery, cCandidatesPtr, C.int(len(candidates)), C.int(maxLength))

	return SimResult{
		Index: int(result.index),
		Score: float32(result.score),
	}
}

// FindMostSimilarDefault finds the most similar text with default max length (512)
func FindMostSimilarDefault(query string, candidates []string) SimResult {
	return FindMostSimilar(query, candidates, 512)
}

// SetMemoryCleanupHandler sets up a finalizer to clean up memory when the Go GC runs
func SetMemoryCleanupHandler() {
	runtime.GC()
}

// IsModelInitialized returns whether the model has been successfully initialized
func IsModelInitialized() bool {
	return modelInitialized
}

// InitClassifier initializes the BERT classifier with the specified model path and number of classes
func InitClassifier(modelPath string, numClasses int, useCPU bool) error {
	var err error
	classifierInitOnce.Do(func() {
		if modelPath == "" {
			// Default to BERT base model if path is empty
			modelPath = "bert-base-uncased"
		}

		if numClasses < 2 {
			err = fmt.Errorf("number of classes must be at least 2, got %d", numClasses)
			return
		}

		fmt.Println("Initializing classifier model:", modelPath)

		// Initialize classifier directly using CGO
		cModelID := C.CString(modelPath)
		defer C.free(unsafe.Pointer(cModelID))

		success := C.init_classifier(cModelID, C.int(numClasses), C.bool(useCPU))
		if !bool(success) {
			err = fmt.Errorf("failed to initialize classifier model")
		}
	})
	return err
}

// InitPIIClassifier initializes the BERT PII classifier with the specified model path and number of classes
func InitPIIClassifier(modelPath string, numClasses int, useCPU bool) error {
	var err error
	piiClassifierInitOnce.Do(func() {
		if modelPath == "" {
			// Default to a suitable PII classification model if path is empty
			modelPath = "./pii_classifier_linear_model"
		}

		if numClasses < 2 {
			err = fmt.Errorf("number of classes must be at least 2, got %d", numClasses)
			return
		}

		fmt.Println("Initializing PII classifier model:", modelPath)

		// Initialize PII classifier directly using CGO
		cModelID := C.CString(modelPath)
		defer C.free(unsafe.Pointer(cModelID))

		success := C.init_pii_classifier(cModelID, C.int(numClasses), C.bool(useCPU))
		if !bool(success) {
			err = fmt.Errorf("failed to initialize PII classifier model")
		}
	})
	return err
}

// InitJailbreakClassifier initializes the BERT jailbreak classifier with the specified model path and number of classes
func InitJailbreakClassifier(modelPath string, numClasses int, useCPU bool) error {
	var err error
	jailbreakClassifierInitOnce.Do(func() {
		if modelPath == "" {
			// Default to the jailbreak classification model if path is empty
			modelPath = "./jailbreak_classifier_linear_model"
		}

		if numClasses < 2 {
			err = fmt.Errorf("number of classes must be at least 2, got %d", numClasses)
			return
		}

		fmt.Println("Initializing jailbreak classifier model:", modelPath)

		// Initialize jailbreak classifier directly using CGO
		cModelID := C.CString(modelPath)
		defer C.free(unsafe.Pointer(cModelID))

		success := C.init_jailbreak_classifier(cModelID, C.int(numClasses), C.bool(useCPU))
		if !bool(success) {
			err = fmt.Errorf("failed to initialize jailbreak classifier model")
		}
	})
	return err
}

// ClassifyText classifies the provided text and returns the predicted class and confidence
func ClassifyText(text string) (ClassResult, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.classify_text(cText)

	if result.class < 0 {
		return ClassResult{}, fmt.Errorf("failed to classify text")
	}

	return ClassResult{
		Class:      int(result.class),
		Confidence: float32(result.confidence),
	}, nil
}

// ClassifyPIIText classifies the provided text for PII detection and returns the predicted class and confidence
func ClassifyPIIText(text string) (ClassResult, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.classify_pii_text(cText)

	if result.class < 0 {
		return ClassResult{}, fmt.Errorf("failed to classify PII text")
	}

	return ClassResult{
		Class:      int(result.class),
		Confidence: float32(result.confidence),
	}, nil
}

// ClassifyJailbreakText classifies the provided text for jailbreak detection and returns the predicted class and confidence
func ClassifyJailbreakText(text string) (ClassResult, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.classify_jailbreak_text(cText)

	if result.class < 0 {
		return ClassResult{}, fmt.Errorf("failed to classify jailbreak text")
	}

	return ClassResult{
		Class:      int(result.class),
		Confidence: float32(result.confidence),
	}, nil
}
