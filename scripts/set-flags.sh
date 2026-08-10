#!/bin/bash
# =============================================================================
# set-flags.sh — Toggle RAG feature flags on the running ECS service
# =============================================================================
# Directly patches the ECS task definition environment variables and forces
# a new deployment — no Docker rebuild needed (~30 seconds vs ~20 minutes).
#
# Recommended off (high latency cost, marginal accuracy gain):
#   USE_LLM_RERANKER=false        saves ~1 LLM call per query
#   USE_CONTEXT_COMPRESSION=false saves ~1 LLM call per query
#   USE_FAITHFULNESS_CHECK=false  saves ~1-2 LLM calls per query
#
# HOW TO RUN:
#   chmod +x scripts/set-flags.sh
#   ./scripts/set-flags.sh
#
# To change the flags, edit the FLAGS section below.
# =============================================================================

set -e

REGION="${AWS_REGION:-ca-central-1}"

# ── Edit these to change which flags are on/off ──────────────────────────────
FLAG_USE_QUERY_ENHANCEMENT="true"
FLAG_USE_HYBRID_RETRIEVAL="true"   # Python BM25 rebuilds from scratch each query — use Weaviate native hybrid instead
FLAG_USE_DOCUMENT_INTELLIGENCE="true"
FLAG_USE_ADAPTIVE_CHUNKING="true"
FLAG_USE_VLM_SUMMARIES="true"
FLAG_USE_LLM_RERANKER="true"
FLAG_USE_CONTEXT_COMPRESSION="true"
FLAG_USE_FAITHFULNESS_CHECK="true"

# Agentic pipeline — off until the eval harness shows it beats the linear baseline
FLAG_USE_LANGGRAPH_AGENT="false"

# Vector store backend
FLAG_PREFER_WEAVIATE="true"         # route queries to Weaviate Cloud (pre-built indexes, fast)
# ─────────────────────────────────────────────────────────────────────────────

echo "Fetching ECS cluster and service names..."
CLUSTER=$(aws cloudformation describe-stacks \
  --stack-name RagAgentApp \
  --query "Stacks[0].Outputs[?OutputKey=='EcsClusterName'].OutputValue" \
  --output text --region "$REGION")

SERVICE=$(aws cloudformation describe-stacks \
  --stack-name RagAgentApp \
  --query "Stacks[0].Outputs[?OutputKey=='EcsServiceName'].OutputValue" \
  --output text --region "$REGION")

echo "Cluster: $CLUSTER"
echo "Service: $SERVICE"

# Get the ARN of the currently active task definition
TASK_DEF_ARN=$(aws ecs describe-services \
  --cluster "$CLUSTER" \
  --services "$SERVICE" \
  --query "services[0].taskDefinition" \
  --output text --region "$REGION")

echo "Current task definition: $TASK_DEF_ARN"

# Download it as JSON
aws ecs describe-task-definition \
  --task-definition "$TASK_DEF_ARN" \
  --query taskDefinition \
  --region "$REGION" \
  > /tmp/task-def.json

# Apply all flag overrides and strip registration-only fields in one Python pass
python3 - <<'PYEOF'
import json

flags = {
    "USE_QUERY_ENHANCEMENT":   "__FLAG_USE_QUERY_ENHANCEMENT__",
    "USE_HYBRID_RETRIEVAL":    "__FLAG_USE_HYBRID_RETRIEVAL__",
    "USE_DOCUMENT_INTELLIGENCE": "__FLAG_USE_DOCUMENT_INTELLIGENCE__",
    "USE_ADAPTIVE_CHUNKING":   "__FLAG_USE_ADAPTIVE_CHUNKING__",
    "USE_VLM_SUMMARIES":       "__FLAG_USE_VLM_SUMMARIES__",
    "USE_LLM_RERANKER":        "__FLAG_USE_LLM_RERANKER__",
    "USE_CONTEXT_COMPRESSION": "__FLAG_USE_CONTEXT_COMPRESSION__",
    "USE_FAITHFULNESS_CHECK":  "__FLAG_USE_FAITHFULNESS_CHECK__",
    "USE_LANGGRAPH_AGENT":     "__FLAG_USE_LANGGRAPH_AGENT__",
    "PREFER_WEAVIATE":         "__FLAG_PREFER_WEAVIATE__",
}

data = json.load(open("/tmp/task-def.json"))
env = data["containerDefinitions"][0]["environment"]

for key, value in flags.items():
    for e in env:
        if e["name"] == key:
            e["value"] = value
            break
    else:
        env.append({"name": key, "value": value})

for field in ["taskDefinitionArn", "revision", "status", "requiresAttributes",
              "compatibilities", "registeredAt", "registeredBy", "deregisteredAt"]:
    data.pop(field, None)

json.dump(data, open("/tmp/task-def-patched.json", "w"), indent=2)
print("Flags patched.")
PYEOF

# Substitute actual shell variable values into the patched JSON
sed -i '' \
  -e "s/__FLAG_USE_QUERY_ENHANCEMENT__/$FLAG_USE_QUERY_ENHANCEMENT/g" \
  -e "s/__FLAG_USE_HYBRID_RETRIEVAL__/$FLAG_USE_HYBRID_RETRIEVAL/g" \
  -e "s/__FLAG_USE_DOCUMENT_INTELLIGENCE__/$FLAG_USE_DOCUMENT_INTELLIGENCE/g" \
  -e "s/__FLAG_USE_ADAPTIVE_CHUNKING__/$FLAG_USE_ADAPTIVE_CHUNKING/g" \
  -e "s/__FLAG_USE_VLM_SUMMARIES__/$FLAG_USE_VLM_SUMMARIES/g" \
  -e "s/__FLAG_USE_LLM_RERANKER__/$FLAG_USE_LLM_RERANKER/g" \
  -e "s/__FLAG_USE_CONTEXT_COMPRESSION__/$FLAG_USE_CONTEXT_COMPRESSION/g" \
  -e "s/__FLAG_USE_FAITHFULNESS_CHECK__/$FLAG_USE_FAITHFULNESS_CHECK/g" \
  -e "s/__FLAG_USE_LANGGRAPH_AGENT__/$FLAG_USE_LANGGRAPH_AGENT/g" \
  -e "s/__FLAG_PREFER_WEAVIATE__/$FLAG_PREFER_WEAVIATE/g" \
  /tmp/task-def-patched.json

echo ""
echo "Registering new task definition revision..."
NEW_ARN=$(aws ecs register-task-definition \
  --cli-input-json file:///tmp/task-def-patched.json \
  --query "taskDefinition.taskDefinitionArn" \
  --output text --region "$REGION")

echo "New task definition: $NEW_ARN"

echo ""
echo "Forcing new ECS deployment with updated flags..."
aws ecs update-service \
  --cluster "$CLUSTER" \
  --service "$SERVICE" \
  --task-definition "$NEW_ARN" \
  --force-new-deployment \
  --region "$REGION" \
  --output text --query "service.taskDefinition" > /dev/null

echo ""
echo "Done. New containers rolling out (~2-3 min)."
echo "Flags applied:"
echo "  USE_QUERY_ENHANCEMENT=$FLAG_USE_QUERY_ENHANCEMENT"
echo "  USE_HYBRID_RETRIEVAL=$FLAG_USE_HYBRID_RETRIEVAL"
echo "  USE_DOCUMENT_INTELLIGENCE=$FLAG_USE_DOCUMENT_INTELLIGENCE"
echo "  USE_ADAPTIVE_CHUNKING=$FLAG_USE_ADAPTIVE_CHUNKING"
echo "  USE_VLM_SUMMARIES=$FLAG_USE_VLM_SUMMARIES"
echo "  USE_LLM_RERANKER=$FLAG_USE_LLM_RERANKER"
echo "  USE_CONTEXT_COMPRESSION=$FLAG_USE_CONTEXT_COMPRESSION"
echo "  USE_FAITHFULNESS_CHECK=$FLAG_USE_FAITHFULNESS_CHECK"
echo "  PREFER_WEAVIATE=$FLAG_PREFER_WEAVIATE"
echo ""
echo "Watch rollout:"
echo "  aws ecs describe-services --cluster $CLUSTER --services $SERVICE --region $REGION --query 'services[0].{running:runningCount,pending:pendingCount,desired:desiredCount}'"
