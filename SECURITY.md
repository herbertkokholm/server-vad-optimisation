# Security Policy

## Supported Versions

This project is currently under active development. Security updates are applied to the latest version only.

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| Older   | :x:                |

## API Keys

sweep.py and adaptive_tuner.py communicate with the OpenAI Realtime API.  
API keys must never be committed to the repository. Use an environment variable instead:

bash export OPENAI_API_KEY="sk-..." 

Add .env files and any key files to .gitignore.

## Session Data

sweep_runs/ contains JSONL files with recording data from clinical simulation sessions. These files:

- must not be published or shared without ethical approval  
- should be deleted locally once analysis is complete  
- are excluded from this repository via .gitignore  

example_data/ is anonymised and contains no personally identifiable information.

## Reporting a Vulnerability

Please open a private [GitHub Security Advisory](../../security/advisories/new) for this repository.  

Do not create a public GitHub issue for security-related findings.