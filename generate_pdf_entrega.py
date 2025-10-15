#!/usr/bin/env python3
"""
Gerador de PDF para entrega do projeto FarmTech YOLO
FIAP - Faculdade de Informática e Administração Paulista
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import HexColor, black, blue
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
import os

def create_delivery_pdf():
    """Cria o PDF de entrega do projeto FarmTech YOLO"""
    
    # Configuração do documento
    filename = "Entrega_FarmTech_YOLO.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4, 
                          rightMargin=2*cm, leftMargin=2*cm,
                          topMargin=2*cm, bottomMargin=2*cm)
    
    # Estilos
    styles = getSampleStyleSheet()
    
    # Estilo personalizado para título
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=HexColor('#1f4e79')
    )
    
    # Estilo para subtítulos
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=20,
        textColor=HexColor('#2f5f8f')
    )
    
    # Estilo para texto normal
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        alignment=TA_JUSTIFY
    )
    
    # Estilo para informações da equipe
    team_style = ParagraphStyle(
        'TeamStyle',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=4,
        leftIndent=20
    )
    
    # Lista de elementos do PDF
    story = []
    
    # Logo da FIAP (se existir)
    logo_path = "assets/logo-fiap.png"
    if os.path.exists(logo_path):
        try:
            logo = Image(logo_path, width=3*inch, height=1.5*inch)
            logo.hAlign = 'CENTER'
            story.append(logo)
            story.append(Spacer(1, 20))
        except:
            pass
    
    # Título principal
    story.append(Paragraph("FIAP - Faculdade de Informática e Administração Paulista", 
                          ParagraphStyle('Center', alignment=TA_CENTER, fontSize=12, spaceAfter=10)))
    
    story.append(Paragraph("FarmTech YOLO: Sistema Inteligente de Detecção de Celulares com Computer Vision", 
                          title_style))
    
    # Informações da equipe
    story.append(Paragraph("👨‍🎓 Integrantes da Equipe", subtitle_style))
    
    team_members = [
        "Gabriella Serni Ponzetta – RM 566296",
        "João Francisco Maciel Albano – RM 565985", 
        "Fernando Ricardo – RM 566501",
        "Gabriel Schuler Barros – RM 564934"
    ]
    
    for member in team_members:
        story.append(Paragraph(f"• {member}", team_style))
    
    story.append(Spacer(1, 15))
    
    # Professores
    story.append(Paragraph("👩‍🏫 Professores", subtitle_style))
    story.append(Paragraph("<b>Tutores:</b>", team_style))
    story.append(Paragraph("• Lucas Gomes Moreira", team_style))
    story.append(Paragraph("• Leonardo Ruiz Orabona", team_style))
    story.append(Paragraph("<b>Coordenador:</b>", team_style))
    story.append(Paragraph("• André Godoi Chiovato", team_style))
    
    story.append(Spacer(1, 20))
    
    # Link do repositório
    story.append(Paragraph("🔗 Repositório do Projeto", subtitle_style))
    github_link = "https://github.com/joao-albano/farm-tech-rede-neural.git"
    story.append(Paragraph(f'<link href="{github_link}" color="blue">{github_link}</link>', normal_style))
    
    story.append(Spacer(1, 20))
    
    # Descrição do projeto
    story.append(Paragraph("📜 Descrição do Projeto", subtitle_style))
    
    description_text = """
    Este projeto implementa um sistema avançado de detecção de celulares utilizando técnicas de Computer Vision 
    e Deep Learning com YOLO (You Only Look Once). O sistema combina modelos pré-treinados YOLOv8 com redes 
    neurais convolucionais customizadas, proporcionando detecção em tempo real com alta precisão para aplicações 
    em segurança, monitoramento e controle de acesso.
    """
    
    story.append(Paragraph(description_text, normal_style))
    
    description_text2 = """
    O projeto utiliza um dataset especializado de imagens de celulares, aplicando técnicas modernas de 
    processamento de imagens e aprendizado profundo. A solução desenvolvida oferece uma alternativa 
    automatizada e eficiente para sistemas de detecção manual, com aplicações práticas em ambientes 
    corporativos, educacionais e de segurança.
    """
    
    story.append(Paragraph(description_text2, normal_style))
    
    story.append(Spacer(1, 20))
    
    # Principais resultados
    story.append(Paragraph("📊 Principais Resultados", subtitle_style))
    
    results_data = [
        ['Métrica', 'Valor'],
        ['Modelo Utilizado', 'YOLOv8n'],
        ['Precisão de Detecção', '> 85%'],
        ['Velocidade de Processamento', 'Tempo Real'],
        ['Dataset', 'Cellphone Detection Dataset'],
        ['Tecnologias', 'Python, PyTorch, OpenCV, YOLO']
    ]
    
    results_table = Table(results_data, colWidths=[3*inch, 2.5*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1f4e79')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 1, black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f8f9fa')])
    ]))
    
    story.append(results_table)
    story.append(Spacer(1, 30))
    
    # Estrutura do projeto
    story.append(Paragraph("📁 Estrutura do Projeto", subtitle_style))
    
    structure_text = """
    O projeto está organizado seguindo as melhores práticas de desenvolvimento, com separação clara entre 
    código-fonte, documentação, assets e configurações. A estrutura inclui notebooks Jupyter para análise 
    interativa, scripts Python para automação, documentação técnica completa e datasets organizados.
    """
    
    story.append(Paragraph(structure_text, normal_style))
    
    story.append(Spacer(1, 30))
    
    # Data de entrega
    current_date = datetime.now().strftime("%d de %B de %Y")
    story.append(Paragraph(f"📅 Data de Entrega: {current_date}", 
                          ParagraphStyle('DateStyle', fontSize=12, alignment=TA_CENTER, 
                                       textColor=HexColor('#666666'))))
    
    # Gerar o PDF
    doc.build(story)
    print(f"✅ PDF gerado com sucesso: {filename}")
    return filename

if __name__ == "__main__":
    try:
        pdf_file = create_delivery_pdf()
        print(f"📄 Arquivo PDF criado: {pdf_file}")
        print("🎯 O PDF está pronto para entrega!")
    except Exception as e:
        print(f"❌ Erro ao gerar PDF: {e}")
        print("💡 Certifique-se de que a biblioteca reportlab está instalada: pip install reportlab")