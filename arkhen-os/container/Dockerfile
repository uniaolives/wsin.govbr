# arkhen-os/container/Dockerfile
# Base: Arch Linux minimal com systemd
FROM archlinux:base-devel

# Configuração básica do Arch no container
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    echo "LANG=en_US.UTF-8" > /etc/locale.conf

# Instala pacotes ESSENCIAIS (kernel do host, não precisamos compilar)
RUN pacman -Syu --noconfirm --needed \
    base \
    systemd \
    systemd-sysvcompat \
    python \
    python-numpy \
    python-pyglet \
    python-pip \
    git \
    sudo \
    nano \
    htop \
    btop \
    && pacman -Scc --noconfirm  # Limpa cache

# Instala o servidor MCP Python mais estável
RUN pip install --no-cache-dir fastmcp

# Configura usuário 'arkhe' (não-root para segurança)
RUN useradd -m -G wheel -s /bin/bash arkhe && \
    echo "arkhe:arkhe" | chpasswd && \
    echo "%wheel ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Diretório da aplicação
RUN mkdir -p /opt/biogenesis && chown -R arkhe:arkhe /opt/biogenesis

# Copia configurações do sistema
COPY config/ /etc/arkhe/

# Habilita nossos serviços customizados
RUN systemctl enable arkhe-daemon && \
    systemctl enable mcp-server.socket

# Configura o systemd para rodar em container
RUN systemctl set-default multi-user.target && \
    systemctl mask systemd-udev-trigger.service

# Ponto de entrada: systemd como PID 1
STOPSIGNAL SIGRTMIN+3
CMD ["/usr/lib/systemd/systemd", "--log-level=info", "--log-target=journal"]
