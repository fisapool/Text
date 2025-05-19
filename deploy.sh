#!/bin/bash

# Exit on error
set -e

# Configuration
APP_NAME="ai-paraphraser"
DROPLET_IP="YOUR_DROPLET_IP"
SSH_KEY="~/.ssh/id_rsa"
DEPLOY_USER="root"
APP_DIR="/opt/$APP_NAME"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting deployment of $APP_NAME...${NC}"

# Build Docker image
echo "Building Docker image..."
docker build -t $APP_NAME:latest .

# Save Docker image
echo "Saving Docker image..."
docker save $APP_NAME:latest | gzip > $APP_NAME.tar.gz

# Copy files to server
echo "Copying files to server..."
scp -i $SSH_KEY $APP_NAME.tar.gz $DEPLOY_USER@$DROPLET_IP:/tmp/
scp -i $SSH_KEY docker-compose.yml $DEPLOY_USER@$DROPLET_IP:$APP_DIR/
scp -i $SSH_KEY .env $DEPLOY_USER@$DROPLET_IP:$APP_DIR/

# Deploy on server
echo "Deploying on server..."
ssh -i $SSH_KEY $DEPLOY_USER@$DROPLET_IP << 'ENDSSH'
    # Load Docker image
    docker load < /tmp/ai-paraphraser.tar.gz
    
    # Navigate to app directory
    cd /opt/ai-paraphraser
    
    # Stop existing containers
    docker-compose down
    
    # Start new containers
    docker-compose up -d
    
    # Clean up
    rm /tmp/ai-paraphraser.tar.gz
    
    # Check if containers are running
    docker-compose ps
ENDSSH

echo -e "${GREEN}Deployment completed successfully!${NC}"

# Clean up local files
rm $APP_NAME.tar.gz 