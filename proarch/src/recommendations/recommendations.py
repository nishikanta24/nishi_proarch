"""
Recommendations Module
Generates actionable cost-saving recommendations using AI-powered analysis.
This refactored module uses LangChain Expression Language (LCEL) and structured
output parsing to provide consistent, high-quality recommendations.
"""

import pandas as pd
import numpy as np
import logging
import os
import re
from typing import Dict, List, Optional
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# LangChain Imports
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from ..ingestion.schema import Billing, Resource

logger = logging.getLogger(__name__)
load_dotenv()


# --- Pydantic Schemas for Structured Data ---

class RecommendationItem(BaseModel):
    """Single recommendation with details (existing model, unchanged)."""
    resource_id: str = Field(description="Resource identifier")
    service: str = Field(description="Azure/AWS service type")
    resource_group: Optional[str] = Field(description="Resource group")
    issue_type: str = Field(description="Type of issue: idle, unit_cost_spike, or missing_tags")
    current_cost: float = Field(description="Current monthly cost")
    estimated_savings: float = Field(description="Estimated monthly savings")
    confidence: str = Field(description="Confidence level: high, medium, low")
    recommendation: str = Field(description="Specific action to take")
    priority: str = Field(description="Priority: critical, high, medium, low")


class RecommendationOutput(BaseModel):
    """Structured recommendation output from the LLM."""
    action: str = Field(description="The specific, actionable recommendation (e.g., 'Delete this VM').")
    reasoning: str = Field(description="A concise explanation for why this action is recommended.")
    confidence_level: str = Field(description="Confidence score for the recommendation: 'high', 'medium', or 'low'.")
    confidence_reasoning: str = Field(description="Justification for the assigned confidence level.")
    priority: str = Field(description="Priority level for the action: 'critical', 'high', 'medium', or 'low'.")


class RecommendationEngine:
    """Generates cost optimization recommendations using heuristics and AI."""

    def __init__(self, engine, request_id: str = "default"):
        self.engine = engine
        self.request_id = request_id
        self.Session = sessionmaker(bind=engine)
        self.llm = None
        self.output_parser = PydanticOutputParser(pydantic_object=RecommendationOutput)

        self._initialize_llm()

        # Setup recommendation chains if LLM is available
        self.idle_chain = self._setup_recommendation_chain(self._get_idle_prompt_template())
        self.unit_cost_chain = self._setup_recommendation_chain(self._get_unit_cost_prompt_template())
        self.tagging_chain = self._setup_recommendation_chain(self._get_tagging_prompt_template())

    def _initialize_llm(self):
        """Initialize LLM using ChatOpenAI with OpenRouter, matching qa_chain.py."""
        model_name = os.getenv("MODEL_NAME", "deepseek/deepseek-r1-0528-qwen3-8b:free")
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            logger.warning(
                "OPENROUTER_API_KEY not set. LLM features will be disabled. Using rule-based fallbacks.",
                extra={'request_id': self.request_id}
            )
            return

        logger.info(f"Initializing LLM with OpenRouter model: '{model_name}'", extra={'request_id': self.request_id})
        try:
            self.llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.3,  # Slightly higher for creative recommendations
                max_tokens=500,   # Recommendations should be concise
                timeout=15
            )
            logger.info("Successfully initialized LLM for recommendation generation.", extra={'request_id': self.request_id})
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}. Using rule-based fallbacks.", extra={'request_id': self.request_id})
            self.llm = None

    def _validate_input(self, text: str, max_length: int = 200) -> str:
        """Sanitize inputs before using in prompts to prevent injection."""
        if not isinstance(text, str):
            return ""
        # Remove characters that could be used for prompt injection
        cleaned_text = re.sub(r'[<>{|}]', '', text)
        # Truncate to max_length
        return cleaned_text[:max_length]

    def _setup_recommendation_chain(self, prompt_template: PromptTemplate):
        """Sets up a generic LCEL chain for generating recommendations."""
        if not self.llm:
            return None
        return (
            RunnablePassthrough()
            | prompt_template
            | self.llm
            | self.output_parser
        )

    # --- Prompt Templates with Few-Shot Examples ---

    def _get_idle_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""You are a FinOps expert. Generate a specific, actionable recommendation for the idle resource.

FEW-SHOT EXAMPLES:
---
Input:
Service: Virtual Machine
Issue: 0% CPU usage for 30 days
Cost: $450

Output:
{{
    "action": "Delete this Virtual Machine immediately.",
    "reasoning": "The resource has shown zero utilization for a full month, indicating it is no longer in use.",
    "confidence_level": "high",
    "confidence_reasoning": "Zero usage over an extended period is a strong indicator of waste.",
    "priority": "critical"
}}
---
Input:
Service: SQL Database
Issue: 2% DTU usage, sporadic connections
Cost: $150

Output:
{{
    "action": "Downsize the database to the next smallest pricing tier.",
    "reasoning": "The resource is severely underutilized and can be run on a cheaper tier without impacting performance.",
    "confidence_level": "high",
    "confidence_reasoning": "Low, consistent usage below 5% indicates over-provisioning.",
    "priority": "high"
}}
---

CURRENT CASE:
Service: {service}
Issue: {issue}
Cost: ${cost}

Generate ONE recommendation. Output as JSON matching the schema.
{format_instructions}""",
            input_variables=["service", "issue", "cost"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()},
        )

    def _get_unit_cost_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""You are a FinOps expert. Analyze the unit cost spike and provide a recommendation.

FEW-SHOT EXAMPLES:
---
Input:
Service: Blob Storage
Change: +45%
Impact: $800

Output:
{{
    "action": "Investigate the storage account configuration for a recent tier change (e.g., from Cool to Hot).",
    "reasoning": "A sudden 45% unit cost increase often points to a change in storage tier or redundancy settings.",
    "confidence_level": "high",
    "confidence_reasoning": "The magnitude of the percentage increase aligns with common pricing differences between tiers.",
    "priority": "high"
}}
---
Input:
Service: API Gateway
Change: +200%
Impact: $1200

Output:
{{
    "action": "Contact the cloud provider to investigate a potential pricing error or undocumented change for this API Gateway.",
    "reasoning": "A 200% increase is highly unusual and suggests a billing anomaly rather than a standard configuration change.",
    "confidence_level": "medium",
    "confidence_reasoning": "While a pricing error is likely, it could also be a complex usage pattern change.",
    "priority": "critical"
}}
---

CURRENT CASE:
Service: {service}
Change: +{change_pct}%
Impact: ${impact}

Generate ONE recommendation. Output as JSON matching the schema.
{format_instructions}""",
            input_variables=["service", "change_pct", "impact"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()},
        )

    def _get_tagging_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""You are a FinOps expert. Recommend an action to fix the resource tagging gap.

FEW-SHOT EXAMPLES:
---
Input:
Service: Kubernetes Cluster
Missing: owner
Cost: $1200

Output:
{{
    "action": "Identify the team responsible for this Kubernetes Cluster and add an 'owner' tag with their cost center.",
    "reasoning": "Tagging is critical for cost allocation and accountability. This untagged resource hinders financial tracking.",
    "confidence_level": "high",
    "confidence_reasoning": "The 'owner' tag is a fundamental requirement for FinOps governance.",
    "priority": "medium"
}}
---
Input:
Service: Lambda Function
Missing: both (owner, env)
Cost: $80

Output:
{{
    "action": "Assign 'owner' and 'env' (e.g., prod, dev) tags to this Lambda function.",
    "reasoning": "Missing both tags prevents cost allocation by team and by environment, reducing visibility.",
    "confidence_level": "high",
    "confidence_reasoning": "Complete tagging is essential for effective cost management.",
    "priority": "low"
}}
---

CURRENT CASE:
Service: {service}
Missing: {missing}
Cost: ${cost}

Generate ONE recommendation. Output as JSON matching the schema.
{format_instructions}""",
            input_variables=["service", "missing", "cost"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()},
        )

    # --- Recommendation Generation Methods ---

    def _generate_recommendation(self, chain, fallback_func, **kwargs) -> Dict:
        """Generic method to invoke an LCEL chain with a fallback."""
        if chain:
            try:
                # Sanitize inputs
                safe_kwargs = {k: self._validate_input(str(v)) for k, v in kwargs.items()}
                result = chain.invoke(safe_kwargs)
                return {
                    "status": "success",
                    "recommendation": result.action,
                    "confidence": result.confidence_level,
                    "priority": result.priority,
                    "reasoning": result.reasoning
                }
            except Exception as e:
                logger.warning(f"LLM chain failed: {e}. Using fallback.", extra={'request_id': self.request_id})
        
        # Fallback if LLM is unavailable or fails
        return fallback_func(**kwargs)

    def _get_idle_fallback(self, service: str, issue: str, cost: float, **kwargs) -> Dict:
        return {
            "status": "fallback",
            "recommendation": f"Review this {service} resource for deletion or downsizing due to being {issue}.",
            "confidence": "low",
            "priority": "medium" if "underutilized" in issue else "high",
            "reasoning": "Fallback rule: idle resources should be reviewed."
        }

    def _get_unit_cost_fallback(self, service: str, change_pct: float, impact: float, **kwargs) -> Dict:
        return {
            "status": "fallback",
            "recommendation": f"Investigate {service} pricing change ({change_pct:+.1f}%). Contact vendor or check pricing tier.",
            "confidence": "low",
            "priority": "high" if impact > 500 else "medium",
            "reasoning": "Fallback rule: significant unit cost spikes require investigation."
        }

    def _get_tagging_fallback(self, service: str, missing: str, cost: float, **kwargs) -> Dict:
        return {
            "status": "fallback",
            "recommendation": f"Add '{missing}' tag(s) to this ${cost:.2f}/month {service} resource for cost accountability.",
            "confidence": "low",
            "priority": "medium" if cost > 100 else "low",
            "reasoning": "Fallback rule: untagged resources must be tagged for governance."
        }

    # --- Core Detection Logic (SQL queries are unchanged) ---

    def detect_idle_resources(self, month: str, usage_threshold: float = 10.0) -> List[Dict]:
        """Detects idle resources. SQL logic is preserved, recommendation generation is updated."""
        logger.info(f"Detecting idle resources for {month}", extra={'request_id': self.request_id})
        try:
            # SQL Query is unchanged
            query = f"""
            WITH resource_stats AS (
                SELECT b.resource_id, b.service, b.resource_group, b.region, SUM(b.cost) as total_cost,
                       SUM(b.usage_qty) as total_usage, r.owner, r.env
                FROM billing b LEFT JOIN resources r ON b.resource_id = r.resource_id
                WHERE b.invoice_month = '{month}' AND b.cost > 0
                GROUP BY b.resource_id, b.service, b.resource_group, b.region, r.owner, r.env
            ), service_avg AS (
                SELECT service, AVG(total_usage) as service_avg_usage FROM resource_stats GROUP BY service
            )
            SELECT rs.*, CASE WHEN sa.service_avg_usage > 0 THEN ((rs.total_usage / sa.service_avg_usage) * 100) ELSE 0 END as usage_percentile
            FROM resource_stats rs LEFT JOIN service_avg sa ON rs.service = sa.service
            WHERE rs.total_cost > 10 ORDER BY rs.total_cost DESC
            """
            df = pd.read_sql(query, self.engine)
            if df.empty: return []

            idle_df = df[(df['usage_percentile'] < usage_threshold) | (df['total_usage'] == 0)]
            recommendations = []

            for _, row in idle_df.head(20).iterrows():
                issue = "completely idle (zero usage)" if row['total_usage'] == 0 else f"underutilized ({row['usage_percentile']:.1f}% of average)"
                cost = float(row['total_cost'])
                estimated_savings = cost * 0.7

                gen_output = self._generate_recommendation(
                    self.idle_chain, self._get_idle_fallback,
                    service=row['service'], issue=issue, cost=cost
                )

                recommendations.append({
                    'resource_id': row['resource_id'], 'service': row['service'],
                    'resource_group': row.get('resource_group', 'unknown'), 'issue_type': 'idle_resource',
                    'current_cost': round(cost, 2), 'estimated_savings': round(estimated_savings, 2),
                    'confidence': gen_output['confidence'], 'recommendation': gen_output['recommendation'],
                    'priority': gen_output['priority']
                })
            
            logger.info(f"✓ Found {len(recommendations)} idle resources.", extra={'request_id': self.request_id})
            return recommendations
        except Exception as e:
            logger.error(f"Error detecting idle resources: {e}", extra={'request_id': self.request_id}, exc_info=True)
            return [] # Return empty list on error, not raise

    def detect_unit_cost_spikes(self, month: str, previous_month: str, threshold_pct: float = 20.0) -> List[Dict]:
        """Detects unit cost spikes. SQL logic is preserved, recommendation generation is updated."""
        logger.info(f"Detecting unit cost spikes for {month}", extra={'request_id': self.request_id})
        try:
            # SQL Query is unchanged
            query = f"""
            WITH current_costs AS (
                SELECT resource_id, service, resource_group, region, AVG(unit_cost) as current_unit_cost, SUM(cost) as current_cost
                FROM billing WHERE invoice_month = '{month}' GROUP BY resource_id, service, resource_group, region
            ), previous_costs AS (
                SELECT resource_id, AVG(unit_cost) as previous_unit_cost, SUM(cost) as previous_cost
                FROM billing WHERE invoice_month = '{previous_month}' GROUP BY resource_id
            )
            SELECT c.*, p.previous_unit_cost, (c.current_cost - p.previous_cost) as cost_impact,
                   ((c.current_unit_cost - p.previous_unit_cost) / p.previous_unit_cost * 100) as unit_cost_change_pct
            FROM current_costs c JOIN previous_costs p ON c.resource_id = p.resource_id
            WHERE p.previous_unit_cost > 0 AND ((c.current_unit_cost - p.previous_unit_cost) / p.previous_unit_cost * 100) > {threshold_pct}
            ORDER BY cost_impact DESC
            """
            df = pd.read_sql(query, self.engine)
            if df.empty: return []

            recommendations = []
            for _, row in df.head(15).iterrows():
                impact = float(row['cost_impact'])
                change_pct = float(row['unit_cost_change_pct'])
                
                gen_output = self._generate_recommendation(
                    self.unit_cost_chain, self._get_unit_cost_fallback,
                    service=row['service'], change_pct=change_pct, impact=impact
                )

                recommendations.append({
                    'resource_id': row['resource_id'], 'service': row['service'],
                    'resource_group': row.get('resource_group', 'unknown'), 'issue_type': 'unit_cost_spike',
                    'current_cost': round(float(row['current_cost']), 2), 'estimated_savings': round(abs(impact), 2),
                    'confidence': gen_output['confidence'], 'recommendation': gen_output['recommendation'],
                    'priority': gen_output['priority']
                })

            logger.info(f"✓ Found {len(recommendations)} unit cost spikes.", extra={'request_id': self.request_id})
            return recommendations
        except Exception as e:
            logger.error(f"Error detecting unit cost spikes: {e}", extra={'request_id': self.request_id}, exc_info=True)
            return []

    def detect_tagging_gaps(self, month: str) -> List[Dict]:
        """Detects tagging gaps. SQL logic is preserved, recommendation generation is updated."""
        logger.info(f"Detecting tagging gaps for {month}", extra={'request_id': self.request_id})
        try:
            # SQL Query is unchanged
            query = f"""
            SELECT b.resource_id, b.service, b.resource_group, SUM(b.cost) as total_cost,
                   CASE WHEN r.owner IS NULL AND r.env IS NULL THEN 'both' WHEN r.owner IS NULL THEN 'owner' ELSE 'environment' END as missing_tags
            FROM billing b LEFT JOIN resources r ON b.resource_id = r.resource_id
            WHERE b.invoice_month = '{month}' AND (r.owner IS NULL OR r.env IS NULL)
            GROUP BY b.resource_id, b.service, b.resource_group, r.owner, r.env
            ORDER BY total_cost DESC
            """
            df = pd.read_sql(query, self.engine)
            if df.empty: return []

            recommendations = []
            for _, row in df.head(20).iterrows():
                cost = float(row['total_cost'])
                
                gen_output = self._generate_recommendation(
                    self.tagging_chain, self._get_tagging_fallback,
                    service=row['service'], missing=row['missing_tags'], cost=cost
                )

                recommendations.append({
                    'resource_id': row['resource_id'], 'service': row['service'],
                    'resource_group': row.get('resource_group', 'unknown'), 'issue_type': 'missing_tags',
                    'current_cost': round(cost, 2), 'estimated_savings': 0.0,
                    'confidence': gen_output['confidence'], 'recommendation': gen_output['recommendation'],
                    'priority': gen_output['priority']
                })

            logger.info(f"✓ Found {len(recommendations)} resources with tagging gaps.", extra={'request_id': self.request_id})
            return recommendations
        except Exception as e:
            logger.error(f"Error detecting tagging gaps: {e}", extra={'request_id': self.request_id}, exc_info=True)
            return []

    # --- Orchestration (Unchanged as requested) ---

    def generate_all_recommendations(self, month: str, previous_month: str = None) -> Dict:
        """
        Generate comprehensive recommendations across all categories.
        Orchestration logic is preserved. Error handling is standardized.
        """
        logger.info(f"Generating comprehensive recommendations for {month}", extra={'request_id': self.request_id})
        start_time = datetime.now()

        # Auto-calculate previous month if not provided
        if not previous_month:
            from dateutil.relativedelta import relativedelta
            try:
                month_dt = datetime.strptime(month, '%Y-%m')
                prev_month_dt = month_dt - relativedelta(months=1)
                previous_month = prev_month_dt.strftime('%Y-%m')
            except ValueError:
                logger.error(f"Invalid month format: {month}. Expected YYYY-MM.", extra={'request_id': self.request_id})
                return {"status": "error", "message": "Invalid month format. Use YYYY-MM."}

        # Run all detection methods, which now return empty lists on error
        idle_recommendations = self.detect_idle_resources(month)
        unit_cost_recommendations = self.detect_unit_cost_spikes(month, previous_month)
        tagging_recommendations = self.detect_tagging_gaps(month)

        all_recommendations = idle_recommendations + unit_cost_recommendations + tagging_recommendations

        total_savings = sum(r.get('estimated_savings', 0) for r in all_recommendations)
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        all_recommendations.sort(key=lambda x: (priority_order.get(x['priority'], 4), -x.get('estimated_savings', 0)))

        summary = {
            'total_recommendations': len(all_recommendations),
            'by_type': {
                'idle_resources': len(idle_recommendations),
                'unit_cost_spikes': len(unit_cost_recommendations),
                'tagging_gaps': len(tagging_recommendations)
            },
            'by_priority': {p: len([r for r in all_recommendations if r['priority'] == p]) for p in priority_order.keys()},
            'total_potential_savings': round(total_savings, 2),
        }

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"✓ Recommendation generation complete in {duration:.3f}s: "
            f"{len(all_recommendations)} recommendations with ${total_savings:,.2f} potential savings",
            extra={'request_id': self.request_id}
        )

        return {
            'status': 'success',
            'message': f"Generated {summary['total_recommendations']} recommendations.",
            'request_id': self.request_id,
            'generation_time_seconds': round(duration, 3),
            'summary': summary,
            'recommendations': all_recommendations[:50]  # Return top 50
        }
