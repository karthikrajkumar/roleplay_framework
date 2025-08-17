#!/bin/bash

# Kubernetes Health Check Script
# Comprehensive health checking for all services

set -euo pipefail

NAMESPACE=${1:-roleplay-platform}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    Kubernetes Health Check                     ║${NC}"
    echo -e "${CYAN}║                   Namespace: ${NAMESPACE}$(printf '%*s' $((19 - ${#NAMESPACE})) '')║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

check_namespace() {
    echo -e "${BLUE}🏠 Checking namespace...${NC}"
    if kubectl get namespace ${NAMESPACE} &>/dev/null; then
        echo -e "   ${GREEN}✓${NC} Namespace '${NAMESPACE}' exists"
    else
        echo -e "   ${RED}✗${NC} Namespace '${NAMESPACE}' not found"
        return 1
    fi
    echo ""
}

check_pods() {
    echo -e "${BLUE}🚀 Checking pod status...${NC}"
    
    # Get all pods in the namespace
    if ! kubectl get pods -n ${NAMESPACE} &>/dev/null; then
        echo -e "   ${RED}✗${NC} Cannot access pods in namespace ${NAMESPACE}"
        return 1
    fi
    
    local pods_output
    pods_output=$(kubectl get pods -n ${NAMESPACE} --no-headers 2>/dev/null || echo "")
    
    if [[ -z "${pods_output}" ]]; then
        echo -e "   ${YELLOW}⚠${NC} No pods found in namespace ${NAMESPACE}"
        return 0
    fi
    
    local total_pods=0
    local running_pods=0
    local failed_pods=0
    
    while IFS= read -r line; do
        if [[ -n "${line}" ]]; then
            local pod_name=$(echo ${line} | awk '{print $1}')
            local ready=$(echo ${line} | awk '{print $2}')
            local status=$(echo ${line} | awk '{print $3}')
            local restarts=$(echo ${line} | awk '{print $4}')
            
            total_pods=$((total_pods + 1))
            
            case "${status}" in
                "Running")
                    if [[ "${ready}" == *"/"* ]]; then
                        local ready_count=$(echo ${ready} | cut -d'/' -f1)
                        local total_count=$(echo ${ready} | cut -d'/' -f2)
                        if [[ "${ready_count}" == "${total_count}" ]]; then
                            echo -e "   ${GREEN}✓${NC} ${pod_name} - ${status} (${ready})"
                            running_pods=$((running_pods + 1))
                        else
                            echo -e "   ${YELLOW}⚠${NC} ${pod_name} - ${status} (${ready}) - Not all containers ready"
                        fi
                    else
                        echo -e "   ${GREEN}✓${NC} ${pod_name} - ${status}"
                        running_pods=$((running_pods + 1))
                    fi
                    ;;
                "Pending"|"ContainerCreating"|"PodInitializing")
                    echo -e "   ${YELLOW}⏳${NC} ${pod_name} - ${status}"
                    ;;
                "CrashLoopBackOff"|"Error"|"Failed"|"ImagePullBackOff"|"ErrImagePull")
                    echo -e "   ${RED}✗${NC} ${pod_name} - ${status} (Restarts: ${restarts})"
                    failed_pods=$((failed_pods + 1))
                    ;;
                *)
                    echo -e "   ${YELLOW}?${NC} ${pod_name} - ${status}"
                    ;;
            esac
        fi
    done <<< "${pods_output}"
    
    echo ""
    echo -e "   📊 Pod Summary: ${total_pods} total, ${GREEN}${running_pods} running${NC}, ${RED}${failed_pods} failed${NC}"
    echo ""
    
    if [[ ${failed_pods} -gt 0 ]]; then
        return 1
    fi
}

check_services() {
    echo -e "${BLUE}🌐 Checking services...${NC}"
    
    local services_output
    services_output=$(kubectl get services -n ${NAMESPACE} --no-headers 2>/dev/null || echo "")
    
    if [[ -z "${services_output}" ]]; then
        echo -e "   ${YELLOW}⚠${NC} No services found in namespace ${NAMESPACE}"
        return 0
    fi
    
    while IFS= read -r line; do
        if [[ -n "${line}" ]]; then
            local service_name=$(echo ${line} | awk '{print $1}')
            local service_type=$(echo ${line} | awk '{print $2}')
            local cluster_ip=$(echo ${line} | awk '{print $3}')
            local external_ip=$(echo ${line} | awk '{print $4}')
            local ports=$(echo ${line} | awk '{print $5}')
            
            if [[ "${cluster_ip}" != "None" ]] && [[ "${cluster_ip}" != "<none>" ]]; then
                echo -e "   ${GREEN}✓${NC} ${service_name} (${service_type}) - ${cluster_ip}:${ports}"
            else
                echo -e "   ${YELLOW}⚠${NC} ${service_name} (${service_type}) - Headless service"
            fi
        fi
    done <<< "${services_output}"
    
    echo ""
}

check_deployments() {
    echo -e "${BLUE}📦 Checking deployments...${NC}"
    
    local deployments_output
    deployments_output=$(kubectl get deployments -n ${NAMESPACE} --no-headers 2>/dev/null || echo "")
    
    if [[ -z "${deployments_output}" ]]; then
        echo -e "   ${YELLOW}⚠${NC} No deployments found in namespace ${NAMESPACE}"
        return 0
    fi
    
    local failed_deployments=0
    
    while IFS= read -r line; do
        if [[ -n "${line}" ]]; then
            local deployment_name=$(echo ${line} | awk '{print $1}')
            local ready=$(echo ${line} | awk '{print $2}')
            local up_to_date=$(echo ${line} | awk '{print $3}')
            local available=$(echo ${line} | awk '{print $4}')
            
            if [[ "${ready}" == "${available}" ]] && [[ "${available}" != "0" ]]; then
                echo -e "   ${GREEN}✓${NC} ${deployment_name} - ${ready} replicas ready"
            else
                echo -e "   ${RED}✗${NC} ${deployment_name} - ${ready} ready, ${available} available"
                failed_deployments=$((failed_deployments + 1))
            fi
        fi
    done <<< "${deployments_output}"
    
    echo ""
    
    if [[ ${failed_deployments} -gt 0 ]]; then
        return 1
    fi
}

check_persistent_volumes() {
    echo -e "${BLUE}💾 Checking persistent volumes...${NC}"
    
    local pvcs_output
    pvcs_output=$(kubectl get pvc -n ${NAMESPACE} --no-headers 2>/dev/null || echo "")
    
    if [[ -z "${pvcs_output}" ]]; then
        echo -e "   ${YELLOW}⚠${NC} No persistent volume claims found"
        echo ""
        return 0
    fi
    
    while IFS= read -r line; do
        if [[ -n "${line}" ]]; then
            local pvc_name=$(echo ${line} | awk '{print $1}')
            local status=$(echo ${line} | awk '{print $2}')
            local volume=$(echo ${line} | awk '{print $3}')
            local capacity=$(echo ${line} | awk '{print $4}')
            
            case "${status}" in
                "Bound")
                    echo -e "   ${GREEN}✓${NC} ${pvc_name} - ${status} (${capacity})"
                    ;;
                "Pending")
                    echo -e "   ${YELLOW}⏳${NC} ${pvc_name} - ${status}"
                    ;;
                *)
                    echo -e "   ${RED}✗${NC} ${pvc_name} - ${status}"
                    ;;
            esac
        fi
    done <<< "${pvcs_output}"
    
    echo ""
}

check_endpoints() {
    echo -e "${BLUE}🔗 Checking service endpoints...${NC}"
    
    # List of services to check endpoints for
    local services=("postgres" "redis" "api-gateway" "user-management" "ai-orchestration" "notification" "analytics" "real-time-communication")
    
    for service in "${services[@]}"; do
        if kubectl get service ${service} -n ${NAMESPACE} &>/dev/null; then
            local endpoints
            endpoints=$(kubectl get endpoints ${service} -n ${NAMESPACE} --no-headers 2>/dev/null | awk '{print $2}' || echo "")
            
            if [[ -n "${endpoints}" ]] && [[ "${endpoints}" != "<none>" ]]; then
                echo -e "   ${GREEN}✓${NC} ${service} - Has endpoints (${endpoints})"
            else
                echo -e "   ${RED}✗${NC} ${service} - No endpoints available"
            fi
        fi
    done
    
    echo ""
}

check_resource_usage() {
    echo -e "${BLUE}📊 Checking resource usage...${NC}"
    
    # Check if metrics-server is available
    if kubectl top nodes &>/dev/null; then
        echo -e "   ${GREEN}✓${NC} Metrics server is available"
        
        # Show pod resource usage
        local pod_metrics
        pod_metrics=$(kubectl top pods -n ${NAMESPACE} --no-headers 2>/dev/null || echo "")
        
        if [[ -n "${pod_metrics}" ]]; then
            echo -e "   📈 Top resource consuming pods:"
            echo "${pod_metrics}" | head -5 | while IFS= read -r line; do
                if [[ -n "${line}" ]]; then
                    local pod_name=$(echo ${line} | awk '{print $1}')
                    local cpu=$(echo ${line} | awk '{print $2}')
                    local memory=$(echo ${line} | awk '{print $3}')
                    echo -e "      ${pod_name}: CPU ${cpu}, Memory ${memory}"
                fi
            done
        fi
    else
        echo -e "   ${YELLOW}⚠${NC} Metrics server not available - cannot show resource usage"
    fi
    
    echo ""
}

perform_health_checks() {
    echo -e "${BLUE}🏥 Performing application health checks...${NC}"
    
    # List of services with their health endpoints
    local services=(
        "api-gateway:8000"
        "user-management:8001"
        "ai-orchestration:8003"
        "notification:8004"
        "analytics:8005"
        "real-time-communication:8006"
    )
    
    for service_port in "${services[@]}"; do
        local service_name=$(echo ${service_port} | cut -d':' -f1)
        local port=$(echo ${service_port} | cut -d':' -f2)
        
        if kubectl get service ${service_name} -n ${NAMESPACE} &>/dev/null; then
            # Try to get the NodePort
            local node_port
            node_port=$(kubectl get service ${service_name} -n ${NAMESPACE} -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "")
            
            if [[ -n "${node_port}" ]]; then
                # Health check via NodePort
                if curl -sf "http://localhost:${node_port}/health" &>/dev/null; then
                    echo -e "   ${GREEN}✓${NC} ${service_name} - Health check passed (port ${node_port})"
                else
                    echo -e "   ${RED}✗${NC} ${service_name} - Health check failed (port ${node_port})"
                fi
            else
                # Port forward and health check
                echo -e "   ${YELLOW}⚠${NC} ${service_name} - No NodePort, skipping health check"
            fi
        else
            echo -e "   ${YELLOW}⚠${NC} ${service_name} - Service not found"
        fi
    done
    
    echo ""
}

print_summary() {
    local exit_code=$1
    
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                         Summary                                ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    
    if [[ ${exit_code} -eq 0 ]]; then
        echo -e "${GREEN}✅ All health checks passed!${NC}"
        echo -e "${GREEN}   Your Kubernetes environment is healthy and ready.${NC}"
    else
        echo -e "${RED}❌ Some health checks failed!${NC}"
        echo -e "${RED}   Please review the issues above and fix them.${NC}"
        echo ""
        echo -e "${YELLOW}💡 Helpful commands:${NC}"
        echo -e "   📜 View logs: ${CYAN}make k8s-logs${NC}"
        echo -e "   🔍 Describe resources: ${CYAN}kubectl describe pods -n ${NAMESPACE}${NC}"
        echo -e "   🚀 Restart deployments: ${CYAN}make k8s-restart${NC}"
    fi
    
    echo ""
}

main() {
    print_header
    
    local overall_status=0
    
    # Run all health checks
    check_namespace || overall_status=1
    check_pods || overall_status=1
    check_deployments || overall_status=1
    check_services
    check_persistent_volumes
    check_endpoints
    check_resource_usage
    perform_health_checks
    
    print_summary ${overall_status}
    
    exit ${overall_status}
}

main "$@"