# Final degree project documentation guide

This folder contains the LaTeX template extracted from `ALFONSO.zip` and prepared for documenting the final degree project:

- Main file: `__memoria.tex`
- Cover information: `Configuracion/_datos_proyecto.tex`
- Chapters: `Unidades/*/info.tex`
- Bibliography: `referencias.bib`
- User manual appendix: `Unidades/Apéndice/manual_de_usuario.tex`

## Project proposal information to retain

- Title: Extending and improving an interactive dashboard for survival analysis of university student dropout.
- Author: Alfonso Munoz Cuesta.
- Supervisor: Dr. Juan Alfonso Lara Torralbo.
- University: Universidad de Cordoba, Escuela Politecnica Superior de Cordoba.
- Degree: Computer Engineering.
- Project type: analysis and resolution of real practical cases in engineering.
- Subject area: analysis, design, and development of web applications.

## Required objectives

The report must demonstrate coverage of the following points:

- Improve the artificial intelligence module to generate interpretations faster.
- Provide multilingual support, including at least Spanish and English.
- Incorporate or handle new variables, including nonbinary variables where appropriate.
- Provide comparative charts and analyses based on Log-Rank and Cox.
- Integrate new survival analysis techniques: Weibull, Exponential, and Random Survival Forest, or justified equivalents.
- Export reports, charts, or results automatically.

## Expected report structure

The project proposal requires at least:

- Introduction and motivation.
- General and specific objectives.
- Background and relationship to the previous final degree project.
- Theoretical framework for survival analysis and university dropout.
- Methodology: data, preprocessing, models, AI, multilingual support, and export.
- Results and implemented improvements.
- Technical development of the application.
- Discussion, limitations, and conclusions.
- Bibliography.
- Appendices with the user manual, code, screenshots, and supplementary material.

## Project proposal schedule

The estimated total is 300 hours:

| Phase | Hours |
| --- | ---: |
| Preparation | 30 |
| Analysis and design | 60 |
| Development and implementation | 120 |
| Testing and validation | 50 |
| Documentation and manuals | 25 |
| Final evaluation and conclusions | 15 |

## Initial state identified

- The template already contains substantial text in the introduction, objectives, background, resources, requirements, and analysis chapters.
- The theoretical framework, constraints, design, implementation, testing, and conclusions chapters require further development.
- `__memoria.tex` was corrected to include `Unidades/11-Implementación/info.tex`, which was omitted because of an incomplete `\input`.
- `anteproyecto_extract.txt` contains text extracted from the PDF for reference to the original requirements and bibliography.

## Recommended next steps

1. Review and complete the theoretical framework using the project's actual techniques: Kaplan-Meier, Cox, Log-Rank, Weibull, Exponential, and Random Survival Forest.
2. Document the actual code architecture using the repository modules (`cargaDataset.py`, `layout.py`, `analysis_callbacks.py`, `pdf_exporter.py`, etc.).
3. Move the existing user manual from `doc/MANUAL DE USUARIO.pdf` into the report appendix, or summarize it in LaTeX.
4. Clean up `referencias.bib`, which contains inherited references unrelated to this project.
5. Add actual application screenshots and exported results when the functionality is complete.
