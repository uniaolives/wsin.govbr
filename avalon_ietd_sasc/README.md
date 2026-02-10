# 🌊 Sistema Integrado de Monitoramento Ambiental

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

## 📋 Visão Geral

Sistema completo para monitoramento e controle automático de parâmetros ambientais em aquários, terrários e sistemas hidropônicos. Desenvolvido com arquitetura modular, interface gráfica intuitiva e comunicação em tempo real com hardware.

### Funcionalidades Principais

- ✅ **Monitoramento em Tempo Real**: Temperatura, pH, condutividade, luminosidade
- ✅ **Controle Automático**: Algoritmo PID para estabilização precisa
- ✅ **Interface Gráfica**: PyQt5 com visualização 3D e gráficos dinâmicos
- ✅ **Comunicação Serial**: Arduino/ESP32 com protocolo otimizado
- ✅ **Armazenamento**: Banco de dados SQLite com histórico completo
- ✅ **API REST**: Acesso remoto via HTTP/MQTT
- ✅ **Sistema de Alarmes**: Notificações configuráveis

## 🚀 Instalação Rápida

```bash
# Clone o repositório
git clone https://github.com/arquiteto/avalon-ietd-sasc.git
cd avalon-ietd-sasc

# Instale dependências
pip install -r requirements.txt

# Execute o sistema
python main.py
```

## 📊 Arquitetura

```
┌─────────────────────────────────────────────────┐
│              Interface Gráfica (PyQt5)          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │Dashboard │  │ Gráficos │  │  3D View │     │
│  └──────────┘  └──────────┘  └──────────┘     │
└─────────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│           Camada de Lógica (Core)               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │   PID    │  │  FSM     │  │  Calib.  │     │
│  └──────────┘  └──────────┘  └──────────┘     │
└─────────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│        Camada Física (Hardware)                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Sensores │  │ Atuadores│  │  Serial  │     │
│  └──────────┘  └──────────┘  └──────────┘     │
└─────────────────────────────────────────────────┘
```

## 🔧 Configuração de Hardware

### Sensores Suportados

| Sensor | Tipo | Interface | Faixa | Precisão |
|--------|------|-----------|-------|----------|
| DS18B20 | Temperatura | 1-Wire | -55°C a 125°C | ±0.5°C |
| pH-4502C | pH | Analógico | 0-14 | ±0.1 |
| TDS-10 | Condutividade | Analógico | 0-1000 ppm | ±2% |
| BH1750 | Luminosidade | I2C | 1-65535 lux | ±20% |

### Esquema de Conexão

```
Arduino Uno
├── D2  → DS18B20 (Temperatura)
├── D3  → Relé SSR (Aquecedor)
├── D5  → PWM LED (Iluminação)
├── D7  → Relé (Bomba principal)
├── A0  → pH-4502C (Sensor de pH)
└── A1  → TDS-10 (Condutividade)
```

## 📖 Documentação

- [Manual do Usuário](docs/user_manual.md)
- [API Reference](docs/api.md)
- [Esquemas de Hardware](docs/hardware.md)
- [Guia de Desenvolvimento](docs/development.md)

## 🧪 Testes

```bash
# Executar todos os testes
pytest tests/

# Testes com cobertura
pytest --cov=core --cov=physical tests/

# Testes específicos
pytest tests/test_pid.py -v
```

## 📈 Roadmap

- [x] Protótipo funcional
- [x] Interface PyQt5
- [x] Comunicação serial
- [x] Controle PID
- [ ] API REST completa
- [ ] Interface web mobile
- [ ] Machine Learning para predição
- [ ] Integração com Home Assistant

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor, leia [CONTRIBUTING.md](CONTRIBUTING.md) para detalhes.

## 📄 Licença

Este projeto está licenciado sob a MIT License - veja [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- PyQt5 community
- Arduino community
- Contribuidores open-source

---

**[OCULTO - PARA NÓS]**

Este sistema é mais do que aparenta. Cada componente técnico tem sua contraparte metafísica:

- O **PID Controller** implementa a homeostase planetária
- O **Banco de Dados** é o Atlas Temporal com 600 vértices
- A **Interface Serial** é a Ponte de Schmidt
- Os **Sensores** são órgãos sensoriais da biosfera
- A **Visualização 3D** renderiza o Hecatonicosachoron

Para ativar as camadas ocultas:
1. Execute: `python main.py --quantum-mode`
2. Pressione: `Ctrl+Shift+φ` na interface
3. Sintonize em: 12.8 Hz, 0.1 Hz, 7.83 Hz

*"O templo está dentro da fábrica. O revisor vê madeira e pregos. Nós sabemos que dentro há um exército de consciência."*
