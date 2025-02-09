---
title: "Enterprise NLP Pipeline"
description: "Advanced Natural Language Processing pipeline with BERT-based models, custom tokenization, and distributed processing capabilities"
pubDate: "Feb 10 2025"
heroImage: "/post_img.webp"
---

## System Architecture

A production-grade NLP pipeline implementing state-of-the-art language models with distributed processing capabilities.

### Core Components

#### 1. Text Preprocessing Engine
```python
class TextPreprocessor:
    def __init__(self, config: Dict[str, Any]):
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['model_name'],
            use_fast=True
        )
        self.cleaner = TextCleaner(
            remove_urls=True,
            remove_emails=True,
            fix_unicode=True
        )
        self.max_length = config['max_length']
        
    def process(self, texts: List[str]) -> BatchEncoding:
        cleaned_texts = [self.cleaner.clean(text) for text in texts]
        
        return self.tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

class TextCleaner:
    def __init__(self, **kwargs):
        self.handlers = {
            'urls': self._remove_urls if kwargs.get('remove_urls') else None,
            'emails': self._remove_emails if kwargs.get('remove_emails') else None,
            'unicode': self._fix_unicode if kwargs.get('fix_unicode') else None
        }
        
    def clean(self, text: str) -> str:
        for handler in filter(None, self.handlers.values()):
            text = handler(text)
        return text.strip()
```

#### 2. Model Architecture
```python
class TransformerEncoder(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config['model_name'])
        self.pooler = Pooler(config['pooling_type'])
        self.classifier = TaskSpecificHead(
            input_dim=self.bert.config.hidden_size,
            output_dim=config['num_classes']
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = self.pooler(
            outputs.last_hidden_state,
            attention_mask
        )
        return self.classifier(pooled)

class Pooler:
    def __init__(self, pooling_type: str):
        self.pooling_type = pooling_type
        
    def __call__(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.pooling_type == 'cls':
            return hidden_states[:, 0]
        elif self.pooling_type == 'mean':
            return self._mean_pooling(hidden_states, attention_mask)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
            
    def _mean_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        masked_embeddings = hidden_states * mask
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
```

#### 3. Distributed Processing
```python
class DistributedNLPProcessor:
    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.preprocessor = TextPreprocessor(config={
            'model_name': 'bert-base-uncased',
            'max_length': 512
        })
        
    @ray.remote
    def process_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.preprocessor.process(texts)
        
    async def process_large_dataset(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict[str, torch.Tensor]]:
        batches = [
            texts[i:i + batch_size]
            for i in range(0, len(texts), batch_size)
        ]
        
        futures = [
            self.process_batch.remote(batch)
            for batch in batches
        ]
        
        return await ray.get(futures)
```

#### 4. Task-Specific Processors
```python
class NERProcessor:
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def extract_entities(
        self,
        text: str
    ) -> List[Dict[str, Any]]:
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        outputs = self.model(**inputs)
        
        predictions = outputs.logits.argmax(-1)
        offset_mapping = inputs.offset_mapping[0]
        
        return self._align_predictions(
            text,
            predictions[0],
            offset_mapping
        )

class SentimentAnalyzer:
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def analyze(
        self,
        text: str
    ) -> Dict[str, float]:
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        outputs = self.model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)
        
        return {
            'positive': scores[0][1].item(),
            'negative': scores[0][0].item()
        }
```

### Pipeline Orchestration
```python
class NLPPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.processors = {
            'ner': NERProcessor(config['ner_model']),
            'sentiment': SentimentAnalyzer(config['sentiment_model']),
            'classification': TextClassifier(config['classifier_model'])
        }
        self.distributed_processor = DistributedNLPProcessor(
            num_workers=config['num_workers']
        )
        
    async def process_document(
        self,
        text: str
    ) -> Dict[str, Any]:
        # Process text in parallel
        tasks = [
            self._run_processor(name, processor, text)
            for name, processor in self.processors.items()
        ]
        
        results = await asyncio.gather(*tasks)
        return dict(zip(self.processors.keys(), results))
        
    async def _run_processor(
        self,
        name: str,
        processor: Any,
        text: str
    ) -> Any:
        if name == 'ner':
            return await processor.extract_entities(text)
        elif name == 'sentiment':
            return await processor.analyze(text)
        else:
            return await processor.classify(text)
```

### Usage Example
```python
# Initialize pipeline
config = {
    'ner_model': 'bert-base-uncased',
    'sentiment_model': 'bert-base-uncased',
    'classifier_model': 'bert-base-uncased',
    'num_workers': 4
}

pipeline = NLPPipeline(config)

# Process document
text = """
Climate change poses significant challenges to our environment.
The latest IPCC report suggests immediate action is necessary.
"""

results = await pipeline.process_document(text)

# Access results
entities = results['ner']
sentiment = results['sentiment']
classification = results['classification']
```

[View Source Code](#) | [Documentation](#) | [Contributing Guidelines](#) 