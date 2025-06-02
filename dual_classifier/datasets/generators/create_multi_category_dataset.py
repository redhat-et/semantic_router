#!/usr/bin/env python3
"""
Create a dataset with more categories for training the dual classifier.
"""

import json
import random

def create_extended_dataset(output_path: str, num_samples: int = 1000):
    """
    Create a dataset with 8 categories instead of 4.
    """
    # Extended categories list
    categories = [
        'technology', 'science', 'politics', 'sports', 
        'business', 'health', 'entertainment', 'education'
    ]
    
    print(f"🎯 Creating dataset with {len(categories)} categories: {categories}")
    
    samples = []
    for i in range(num_samples):
        category = random.choice(categories)
        
        # Create sample texts with potential PII
        base_texts = {
            'technology': [
                "How does artificial intelligence work?",
                "The latest smartphone features are impressive",
                "Cloud computing is transforming businesses",
                "Cybersecurity threats are increasing",
                "Machine learning algorithms are complex",
                "Blockchain technology enables secure transactions",
                "Virtual reality creates immersive experiences",
                "IoT devices connect our homes"
            ],
            'science': [
                "Climate change affects global temperatures",
                "DNA research reveals new insights",
                "Space exploration continues to advance",
                "Medical breakthroughs save lives",
                "Quantum physics explains reality",
                "Ocean currents influence weather patterns",
                "Renewable energy sources reduce emissions",
                "Genetic engineering offers new possibilities"
            ],
            'politics': [
                "Election results vary by region",
                "Policy changes affect citizens",
                "Government spending increases annually",
                "International relations remain complex",
                "Political debates shape public opinion",
                "Legislative processes require collaboration",
                "Diplomatic negotiations seek peace",
                "Constitutional rights protect freedoms"
            ],
            'sports': [
                "The championship game was exciting",
                "Athletes train for months",
                "Team performance exceeded expectations",
                "Sports statistics reveal trends",
                "Coaching strategies influence outcomes",
                "Olympic records inspire future generations",
                "Professional leagues attract global audiences",
                "Youth sports programs build character"
            ],
            'business': [
                "Market trends affect stock prices",
                "Company profits increased quarterly",
                "Economic indicators show growth",
                "Investment strategies vary widely",
                "Business partnerships drive success",
                "Startup companies disrupt industries",
                "Corporate social responsibility matters",
                "Supply chain management ensures efficiency"
            ],
            'health': [
                "Regular exercise improves cardiovascular health",
                "Balanced nutrition supports overall wellness",
                "Mental health awareness reduces stigma",
                "Preventive care catches problems early",
                "Medical research advances treatment options",
                "Healthcare systems adapt to changing needs",
                "Wellness programs promote healthy lifestyles",
                "Telemedicine expands access to care"
            ],
            'entertainment': [
                "Movies transport audiences to new worlds",
                "Music concerts create memorable experiences",
                "Video games engage players for hours",
                "Television shows tell compelling stories",
                "Streaming platforms offer endless content",
                "Theater performances showcase artistic talent",
                "Celebrity news captures public attention",
                "Social media influencers shape trends"
            ],
            'education': [
                "Students learn through interactive experiences",
                "Teachers adapt to diverse learning styles",
                "Online courses expand educational access",
                "Research universities advance knowledge",
                "Educational technology enhances learning",
                "Curriculum development meets changing needs",
                "Student assessment measures progress",
                "Lifelong learning becomes increasingly important"
            ]
        }
        
        text = random.choice(base_texts[category])
        
        # Occasionally add PII for testing
        if random.random() < 0.3:
            pii_additions = [
                " Contact John Smith at john.smith@company.com",
                " Call 555-123-4567 for more information",
                " Visit our office at 123 Main Street, New York",
                " Email support@business.com for help",
                " Reach out to Sarah Johnson for details",
                " Send a message to info@organization.org",
                " Phone: (555) 987-6543 for appointments",
                " Address: 456 Oak Avenue, Los Angeles, CA"
            ]
            text += random.choice(pii_additions)
        
        sample = {
            'text': text,
            'category': category
        }
        
        samples.append(sample)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    # Print statistics
    category_counts = {}
    for sample in samples:
        cat = sample['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"✅ Created dataset with {num_samples} samples at {output_path}")
    print(f"📊 Category distribution:")
    for cat, count in sorted(category_counts.items()):
        print(f"   {cat}: {count} samples")
    
    return len(categories)

if __name__ == "__main__":
    print("🚀 Creating Extended Multi-Category Dataset")
    print("=" * 50)
    
    # Create training dataset
    train_categories = create_extended_dataset("extended_train_dataset.json", 800)
    
    # Create validation dataset
    val_categories = create_extended_dataset("extended_val_dataset.json", 200)
    
    print(f"\n🎉 Dataset creation complete!")
    print(f"📂 Total categories: {train_categories}")
    print(f"📄 Training samples: 800 (extended_train_dataset.json)")
    print(f"📄 Validation samples: 200 (extended_val_dataset.json)")
    print(f"\n💡 To use these datasets, rename them to:")
    print(f"   mv extended_train_dataset.json real_train_dataset.json")
    print(f"   mv extended_val_dataset.json real_val_dataset.json")
    print(f"\n🚀 Then run your enhanced_trainer.py again!") 