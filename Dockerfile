FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 AS base
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12

FROM base AS python-base

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        vim \
        curl \
        git \
        ca-certificates \
        libssl-dev \
        python3-dev \
        sudo \
        tmux

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN apt update \
    && apt install openssh-server -y
RUN echo 'root:password' | chpasswd \
    && echo "Port 22" >> /etc/ssh/sshd_config \
    && echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config \
    && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

RUN --mount=type=secret,id=PASSWD,env=PASSWD \
    echo "root:$PASSWD" | sudo chpasswd

WORKDIR /FinanceRAG

RUN echo '. "/FinanceRAG/.venv/bin/activate" 2>/dev/null' >> /root/.profile \
    && echo 'cd /FinanceRAG' >> /root/.profile

RUN echo 'source "$HOME/.bashrc" ' >> /root/.bash_profile \
    && echo 'cd /FinanceRAG' >> /root/.bash_profile

RUN echo '. "/FinanceRAG/.venv/bin/activate" 2>/dev/null' >> /root/.bashrc

CMD bash -c "/etc/init.d/ssh restart & /bin/bash"
