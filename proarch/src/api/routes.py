"""
API Routes Module
Implements all FastAPI endpoints for the Cost Insights Copilot
"""

import os
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from .dependencies import get_retriever, get_qa_copilot, get_recommendation_engine
from ..ai.retriever import Retriever
from ..ai.qa_chain import QACopilot
from ..recommendations.recommendations import RecommendationEngine
from ..transformations.kpis import get_available_months

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class AskRequest(BaseModel):
    """Request model for Q&A endpoint"""
    question: str = Field(..., description="Natural language question about cloud costs")
    

class RecommendationsRequest(BaseModel):
    """Request model for recommendations endpoint"""
    month: Optional[str] = Field(None, description="Month in YYYY-MM format")


# ============================================================================
# Endpoint 1: GET /kpi
# ============================================================================

@router.get("/kpi", response_model=Dict[str, Any])
async def get_kpi(
    month: Optional[str] = Query(None, regex=r"^\d{4}-\d{2}$", description="Month in YYYY-MM format (e.g., 2024-05)"),
    retriever: Retriever = Depends(get_retriever)
):
    """
    Get Key Performance Indicators for a specific month.
    
    Returns comprehensive KPI metrics including:
    - Total cost
    - Month-over-month change
    - Top services by cost
    - Top resources
    - Tagging compliance
    - Cost breakdown
    
    If month is not provided, returns KPIs for the most recent available month.
    """
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    logger.info(
        f"GET /kpi request received (month={month})",
        extra={'request_id': request_id}
    )
    
    try:
        # If month not provided, get latest available month
        if not month:
            logger.info(
                "Month not provided, fetching latest available month",
                extra={'request_id': request_id}
            )
            available_months = get_available_months(
                retriever.engine,
                request_id=request_id
            )
            
            if not available_months:
                raise HTTPException(
                    status_code=404,
                    detail="No billing data available in the database"
                )
            
            month = available_months[0]
            logger.info(
                f"Using latest available month: {month}",
                extra={'request_id': request_id}
            )
        
        # Validate month exists in database
        available_months = get_available_months(
            retriever.engine,
            request_id=request_id
        )
        
        if month not in available_months:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for month {month}. Available months: {', '.join(available_months)}"
            )
        
        # Calculate KPIs using the retriever's KPI calculator
        logger.info(
            f"Calculating KPIs for month: {month}",
            extra={'request_id': request_id}
        )
        
        kpis = retriever.kpi_calculator.calculate_monthly_kpis(month)
        
        # Check if KPIs calculation returned no data
        if kpis.get('status') == 'no_data':
            raise HTTPException(
                status_code=404,
                detail=kpis.get('message', f'No data found for {month}')
            )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Add metadata to response
        response = {
            "status": "success",
            "request_id": request_id,
            "month": month,
            "kpis": kpis,
            "metadata": {
                "processing_time_seconds": round(processing_time, 3),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        logger.info(
            f"✓ KPI request completed in {processing_time:.3f}s",
            extra={'request_id': request_id}
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error processing KPI request: {str(e)}",
            extra={'request_id': request_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# ============================================================================
# Endpoint 2: POST /ask (RAG Q&A)
# ============================================================================

@router.post("/ask", response_model=Dict[str, Any])
async def ask_question(
    request: AskRequest,
    retriever: Retriever = Depends(get_retriever),
    qa_copilot: QACopilot = Depends(get_qa_copilot)
):
    """
    Ask a natural language question about cloud costs.
    
    This endpoint implements the full RAG pipeline:
    1. Retrieval: Searches vector store + queries database for relevant context
    2. Generation: Uses LLM to synthesize answer from retrieved context
    3. Returns structured response with answer, sources, and actionable suggestions
    
    Security: Includes prompt injection detection.
    """
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    question = request.question
    
    logger.info(
        f"POST /ask request received: '{question[:100]}...'",
        extra={'request_id': request_id}
    )
    
    try:
        # STEP 0: Security - Check for prompt injection
        logger.info(
            "Checking for prompt injection",
            extra={'request_id': request_id}
        )
        
        is_injection = qa_copilot._check_prompt_injection(question)
        
        if is_injection:
            logger.warning(
                f"Prompt injection detected in question: '{question[:100]}...'",
                extra={'request_id': request_id}
            )
            raise HTTPException(
                status_code=400,
                detail="Security violation: Potential prompt injection detected. Please rephrase your question."
            )
        
        # STEP 1: Retrieval - Get context from vector store + database
        logger.info(
            "STEP 1: Retrieving context",
            extra={'request_id': request_id}
        )
        
        retrieval_result = retriever.retrieve(
            question=question,
            k=5,
            include_structured_data=True
        )
        
        context = retrieval_result['context']
        
        # Check if context is empty or insufficient
        if not context or len(context.strip()) < 50:
            logger.warning(
                "Insufficient context retrieved for question",
                extra={'request_id': request_id}
            )
            raise HTTPException(
                status_code=404,
                detail="Unable to find relevant information to answer your question. Please try rephrasing or asking about a different topic."
            )
        
        logger.info(
            f"Context retrieved: {len(context)} characters, "
            f"{retrieval_result['semantic_results']['documents_retrieved']} docs, "
            f"{retrieval_result['structured_results']['data_points']} data points",
            extra={'request_id': request_id}
        )
        
        # STEP 2: Generation - Use LLM to generate answer
        logger.info(
            "STEP 2: Generating answer with LLM",
            extra={'request_id': request_id}
        )
        
        qa_response = qa_copilot.answer_question(
            question=question,
            context=context,
            request_id=request_id
        )
        
        # Check if QA response indicates an error
        if qa_response.get('status') == 'error':
            raise HTTPException(
                status_code=500,
                detail=qa_response.get('message', 'Failed to generate answer')
            )
        
        # STEP 3: Format response
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = {
            "status": "success",
            "request_id": request_id,
            "question": question,
            "answer": qa_response.get("answer_text", ""),
            "sources": qa_response.get("sources", []),
            "suggestions": qa_response.get("suggestions", []),
            "data_table": qa_response.get("data_table"),
            "token_usage": qa_response.get("token_usage", {}),
            "metadata": {
                "processing_time_seconds": round(processing_time, 3),
                "retrieval_time_seconds": retrieval_result['retrieval_time_seconds'],
                "documents_retrieved": retrieval_result['semantic_results']['documents_retrieved'],
                "data_points_used": retrieval_result['structured_results']['data_points'],
                "intent_classified": retrieval_result['metadata']['intent'],
                "cache_metrics": retrieval_result.get('cache_metrics', {}),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        logger.info(
            f"✓ Q&A request completed in {processing_time:.3f}s",
            extra={'request_id': request_id}
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error processing Q&A request: {str(e)}",
            extra={'request_id': request_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# ============================================================================
# Endpoint 3: POST /recommendations
# ============================================================================

@router.post("/recommendations", response_model=Dict[str, Any])
async def get_recommendations(
    request: RecommendationsRequest,
    retriever: Retriever = Depends(get_retriever),
    recommendation_engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Generate AI-powered cost optimization recommendations.
    
    Analyzes billing data and generates actionable recommendations for:
    - Idle/underutilized resources
    - Unit cost spikes
    - Missing tags causing cost allocation issues
    
    If month is not provided, uses the most recent available month.
    """
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    month = request.month
    
    logger.info(
        f"POST /recommendations request received (month={month})",
        extra={'request_id': request_id}
    )
    
    try:
        # If month not provided, get latest available month
        if not month:
            logger.info(
                "Month not provided, fetching latest available month",
                extra={'request_id': request_id}
            )
            available_months = get_available_months(
                retriever.engine,
                request_id=request_id
            )
            
            if not available_months:
                raise HTTPException(
                    status_code=404,
                    detail="No billing data available in the database"
                )
            
            month = available_months[0]
            logger.info(
                f"Using latest available month: {month}",
                extra={'request_id': request_id}
            )
        
        # Validate month exists in database
        available_months = get_available_months(
            retriever.engine,
            request_id=request_id
        )
        
        if month not in available_months:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for month {month}. Available months: {', '.join(available_months)}"
            )
        
        # Generate recommendations
        logger.info(
            f"Generating recommendations for month: {month}",
            extra={'request_id': request_id}
        )
        
        recommendations = recommendation_engine.generate_all_recommendations(month)

        # Check if recommendations generation had any issues
        if recommendations.get('status') != 'success':
            raise HTTPException(
                status_code=500,
                detail="Failed to generate recommendations"
            )

        # If no recommendations are found, return appropriate message
        if not recommendations.get("recommendations") or len(recommendations.get("recommendations", [])) == 0:
            response = {
                "status": "success",
                "request_id": request_id,
                "month": month,
                "message": "No optimization opportunities found for this month. Your resources appear to be well-optimized.",
                "recommendations": {
                    "status": "success",
                    "summary": {
                        "total_recommendations": 0,
                        "total_potential_savings": 0,
                        "by_type": {
                            "idle_resources": 0,
                            "unit_cost_spikes": 0,
                            "tagging_gaps": 0
                        }
                    },
                    "recommendations": []
                },
                "metadata": {
                    "total_recommendations": 0,
                    "estimated_savings": 0,
                    "processing_time_seconds": round((datetime.now() - start_time).total_seconds(), 3),
                    "timestamp": datetime.now().isoformat()
                }
            }
            return response
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Add metadata to response
        response = {
            "status": "success",
            "request_id": request_id,
            "month": month,
            "recommendations": recommendations,
            "metadata": {
                "total_recommendations": recommendations.get('summary', {}).get('total_recommendations', 0),
                "estimated_savings": recommendations.get('summary', {}).get('total_potential_savings', 0),
                "processing_time_seconds": round(processing_time, 3),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        logger.info(
            f"✓ Recommendations request completed in {processing_time:.3f}s",
            extra={'request_id': request_id}
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error processing recommendations request: {str(e)}",
            extra={'request_id': request_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# ============================================================================
# Health Check Endpoint
# ============================================================================

@router.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "AI Cost & Insights Copilot API"
    }