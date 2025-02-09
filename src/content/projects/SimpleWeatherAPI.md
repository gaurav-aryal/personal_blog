---
title: "Simple Weather API Implementation"
description: "A comprehensive guide to building a weather data service using FastAPI, with caching and error handling"
pubDate: "Mar 23 2023"
heroImage: "/post_img.webp"
---

## Technical Implementation Guide

This guide details the implementation of a weather data service using FastAPI and the OpenWeatherMap API, featuring caching, rate limiting, and comprehensive error handling.

> **Source Code**: Find the complete implementation at [SimpleWeatherAPI](https://github.com/gaurav-aryal/SimpleWeatherAPI)

### Prerequisites

1. **Development Environment**
   - Python 3.8+
   - FastAPI framework
   - Redis for caching
   - uvicorn ASGI server
   - pydantic for data validation

2. **API Requirements**
   - OpenWeatherMap API key
   - Rate limit configuration
   - SSL certificate (for production)

### Installation Steps

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install fastapi uvicorn[standard] redis requests python-dotenv
```

### Core Components Implementation

1. **API Configuration**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Weather API Service",
    description="RESTful API for weather data with caching",
    version="1.0.0"
)

WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
CACHE_EXPIRATION = 300  # 5 minutes
```

2. **Data Models**
```python
class WeatherResponse(BaseModel):
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: int = Field(..., description="Humidity percentage")
    wind_speed: float = Field(..., description="Wind speed in m/s")
    description: str = Field(..., description="Weather description")
    
    class Config:
        schema_extra = {
            "example": {
                "temperature": 22.5,
                "humidity": 65,
                "wind_speed": 5.2,
                "description": "scattered clouds"
            }
        }
```

3. **Caching Implementation**
```python
import redis
from datetime import timedelta

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True
)

async def get_cached_weather(location: str) -> Optional[Dict[str, Any]]:
    try:
        cached_data = redis_client.get(f"weather:{location}")
        return json.loads(cached_data) if cached_data else None
    except redis.RedisError as e:
        logger.error(f"Redis error: {str(e)}")
        return None
```

### API Endpoints Implementation

```python
@app.get("/weather/{city}", response_model=WeatherResponse)
async def get_weather(city: str, units: str = "metric"):
    try:
        # Check cache first
        cached_data = await get_cached_weather(city)
        if cached_data:
            return WeatherResponse(**cached_data)
        
        # Fetch from OpenWeatherMap if not cached
        weather_data = await fetch_weather_data(city, units)
        
        # Cache the results
        await cache_weather_data(city, weather_data)
        
        return WeatherResponse(**weather_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
```

### Error Handling

```python
class WeatherAPIException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail

@app.exception_handler(WeatherAPIException)
async def weather_exception_handler(request, exc: WeatherAPIException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )
```

### Rate Limiting Implementation

```python
from fastapi import Request
from fastapi.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timedelta

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: FastAPI,
        calls: int = 60,
        period: int = 60
    ):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.cache = {}

    async def dispatch(
        self,
        request: Request,
        call_next
    ):
        client_ip = request.client.host
        now = datetime.now()
        
        if client_ip in self.cache:
            calls = self.cache[client_ip]
            if len(calls) >= self.calls:
                oldest_call = calls[0]
                if now - oldest_call < timedelta(seconds=self.period):
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded"
                    )
                calls.pop(0)
        else:
            self.cache[client_ip] = []
            
        self.cache[client_ip].append(now)
        return await call_next(request)
```

### Data Validation

```python
from pydantic import validator

class WeatherRequest(BaseModel):
    city: str = Field(..., min_length=1, max_length=100)
    units: str = Field("metric", regex="^(metric|imperial)$")
    
    @validator('city')
    def validate_city(cls, v):
        if not v.replace(" ", "").isalpha():
            raise ValueError("City name must contain only letters")
        return v.title()
```

### Deployment Configuration

```python
# uvicorn configuration
config = {
    "app": "main:app",
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "reload": False,
    "ssl_keyfile": "/path/to/key.pem",
    "ssl_certfile": "/path/to/cert.pem"
}

# Docker configuration
```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Testing Implementation

```python
from fastapi.testclient import TestClient
import pytest

@pytest.fixture
def client():
    return TestClient(app)

def test_get_weather(client):
    response = client.get("/weather/London")
    assert response.status_code == 200
    data = response.json()
    assert "temperature" in data
    assert "humidity" in data
```

### Important Notes

1. **API Security**
   - Implement API key authentication
   - Use HTTPS in production
   - Sanitize user inputs
   - Rate limit by IP/API key

2. **Performance Optimization**
   - Implement response compression
   - Use connection pooling
   - Optimize cache usage
   - Monitor response times

For detailed implementation and updates, visit the [GitHub Repository](https://github.com/gaurav-aryal/SimpleWeatherAPI).