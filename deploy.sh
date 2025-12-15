#!/bin/bash
# =============================================================================
# Einstein - Electricity Theft Detection System
# Deployment Script
# =============================================================================

set -e

echo "⚡ Einstein Theft Detection System - Deployment"
echo "================================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/infra/docker-compose.yml"

# Parse arguments
COMMAND=${1:-up}

case $COMMAND in
  up)
    echo -e "${GREEN}🚀 Starting all services...${NC}"
    docker-compose -f "$COMPOSE_FILE" up -d
    
    echo -e "\n${GREEN}✅ Services started!${NC}"
    echo "================================================"
    echo "🌐 Frontend:  http://localhost:3000"
    echo "📡 Backend:   http://localhost:8000"
    echo "📚 API Docs:  http://localhost:8000/docs"
    echo "🔍 Nginx:     http://localhost:80"
    ;;
    
  down)
    echo -e "${YELLOW}⏹️ Stopping all services...${NC}"
    docker-compose -f "$COMPOSE_FILE" down
    echo -e "${GREEN}✅ Services stopped${NC}"
    ;;
    
  restart)
    echo -e "${YELLOW}🔄 Restarting all services...${NC}"
    docker-compose -f "$COMPOSE_FILE" restart
    echo -e "${GREEN}✅ Services restarted${NC}"
    ;;
    
  build)
    echo -e "${GREEN}🔨 Building all images...${NC}"
    docker-compose -f "$COMPOSE_FILE" build --no-cache
    echo -e "${GREEN}✅ Build complete${NC}"
    ;;
    
  logs)
    echo -e "${GREEN}📋 Showing logs...${NC}"
    docker-compose -f "$COMPOSE_FILE" logs -f
    ;;
    
  status)
    echo -e "${GREEN}📊 Service status:${NC}"
    docker-compose -f "$COMPOSE_FILE" ps
    ;;
    
  clean)
    echo -e "${RED}🧹 Cleaning up...${NC}"
    docker-compose -f "$COMPOSE_FILE" down -v --rmi local
    echo -e "${GREEN}✅ Cleanup complete${NC}"
    ;;
    
  *)
    echo "Usage: $0 {up|down|restart|build|logs|status|clean}"
    echo ""
    echo "Commands:"
    echo "  up      - Start all services"
    echo "  down    - Stop all services"
    echo "  restart - Restart all services"
    echo "  build   - Build Docker images"
    echo "  logs    - Show service logs"
    echo "  status  - Show service status"
    echo "  clean   - Remove containers and images"
    exit 1
    ;;
esac

# =============================================================================
# Cloud Deployment Notes
# =============================================================================
cat << 'EOF'

================================================================================
☁️ CLOUD DEPLOYMENT OPTIONS
================================================================================

📦 BACKEND (Railway/Render)
---------------------------
1. Connect your GitHub repository
2. Set root directory: src/backend
3. Set build command: pip install -r requirements.txt
4. Set start command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
5. Add environment variable: ARTIFACTS_DIR=/app/artifacts

🌐 FRONTEND (Vercel)
--------------------
1. Connect your GitHub repository
2. Set root directory: src/frontend
3. Framework preset: Next.js
4. Add environment variable: NEXT_PUBLIC_API_URL=https://your-backend-url.railway.app

🐳 DOCKER HUB
-------------
# Build and push images
docker build -t yourusername/einstein-backend:latest ./src/backend
docker push yourusername/einstein-backend:latest

docker build -t yourusername/einstein-frontend:latest ./src/frontend
docker push yourusername/einstein-frontend:latest

================================================================================
EOF
