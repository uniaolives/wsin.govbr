#!/bin/bash
# arkhen-os/build.sh
set -e  # Sai no primeiro erro

echo "🔨 Construindo Arkhe(n) Container OS..."

# 1. Build da imagem Docker
docker build -t arkhen-os:latest -f container/Dockerfile .

# 2. Cria diretório compartilhado para o código Bio-Gênese
mkdir -p shared/biogenesis

echo "✅ Imagem construída: arkhen-os:latest"
echo ""
echo "🚀 PARA EXECUTAR (escolha uma opção):"
echo ""
echo "OPÇÃO 1: Docker simples (apenas bash):"
echo "  docker run -it --rm --name arkhen --cap-add=SYS_ADMIN arkhen-os:latest"
echo ""
echo "OPÇÃO 2: systemd-nspawn (container completo com systemd):"
echo "  sudo systemd-nspawn --boot --directory=/var/lib/machines/arkhen \\"
echo "    --bind=$(pwd)/shared/biogenesis:/opt/biogenesis \\"
echo "    --capability=all \\"
echo "    --network-bridge=br0"
echo ""
echo "OPÇÃO 3: Docker com bind mount do seu código:"
echo "  docker run -it --rm --name arkhen \\"
echo "    --cap-add=SYS_ADMIN \\"
echo "    -v $(pwd)/shared/biogenesis:/opt/biogenesis \\"
echo "    -v /sys/fs/cgroup:/sys/fs/cgroup:ro \\"
echo "    arkhen-os:latest"
echo ""
echo "📝 Após iniciar, dentro do container:"
echo "  1. sudo systemctl start arkhe-daemon"
echo "  2. sudo systemctl start mcp-server.socket"
echo "  3. Conecte-se via MCP no socket: /run/mcp-server/mcp.sock"
