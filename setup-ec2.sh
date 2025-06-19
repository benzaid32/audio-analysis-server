#!/bin/bash
# Enterprise-grade Audio Analysis Server EC2 Setup Script
# Production deployment for Virtuoso AI Music Lab

# Update system packages
echo "Updating system packages..."
sudo yum update -y

# Install essential tools
echo "Installing Git, Docker, and dependencies..."
sudo yum install -y git docker

# Enable and start Docker service
sudo systemctl enable docker
sudo systemctl start docker

# Add ec2-user to docker group
sudo usermod -aG docker ec2-user
echo "You'll need to log out and log back in for group changes to take effect"

# Install Docker Compose
echo "Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installations
echo "Verifying installations..."
docker --version
docker-compose --version

# Clone repository
echo "Cloning repository..."
git clone https://github.com/benzaid32/audio-analysis-server.git
cd audio-analysis-server

# Create logs directory
mkdir -p logs

# Deploy with docker-compose
echo "Deploying audio analysis server..."
docker-compose up -d --build

# Print status and next steps
echo ""
echo "-----------------------------------------------"
echo "Deployment Status:"
docker ps
echo ""
echo "Next Steps:"
echo "1. Configure Security Group to allow access on port 8000"
echo "2. Set AUDIO_ANALYSIS_SERVER_URL in Supabase to:"
echo "   http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000"
echo "3. Test the server with: curl http://localhost:8000/health"
echo "-----------------------------------------------"
