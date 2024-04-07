import os
import re
import math
import numpy as np
import re
from typing import List, Tuple, Set

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import PyPDF2
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import nltk
import copy

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import resolve1

from tqdm import tqdm


_DEFAULT_WEIGHTS = {
    "positional": {
        "total": 0.5,    # Вес всех геометрических аргументов
        "length": 0.45,
        "x0": 0.1,
        "x1": 0.15,
        "width": 0.1,
        "pagenum": 0.2,
    },
    "semantical": {
        "total": 0.5,    # Вес всех семантических аргументов
        "regex": 0.4,
        "verbs": 0.3,
        "linebreaks": 0.3 
    }
}

_DEFAULT_LEN_SLOPES = (2, 0.4)


def extract_style(fontname: str) -> Tuple[str, str]:
    STYLES = ['BoldItalic', 'Standart', 'Bold', 'Italic']
    pattern = '|'.join(STYLES)
    
    if not isinstance(fontname, str):
        raise TypeError(f'fontname should be a string, got {type(fontname)} type')
    
    if style_search_res := re.search(pattern, fontname):
        style = style_search_res.group(0)
        font = fontname.split(style)[0]
        # Truncate last symbol because it's dash or comma
        return font[:-1], style
    return fontname, 'Standart'
        

def get_text_format(element) -> Tuple[str, int]:
    """
    Функция итерируется по каждому символу в элементе.
    Для каждого элемента фиксирует шрифт и размер и возвращает самые частовстречающиеся в тексте.
    """
    if not isinstance(element, LTTextContainer):
        raise TypeError('Can only get text format for LTTextContainer elements')
        
    if element.is_empty():
        raise ValueError('Textbox is empty!')
    
    fonts = dict()
    sizes = dict()
    styles = dict()
    
    
    for line in element:
        if isinstance(line, LTTextContainer):
            for character in line:
                if isinstance(character, LTChar) and isinstance(character.fontname, str):
                    character.size = round(character.size)
                    font, style = extract_style(character.fontname)
                    
                    count_font = fonts.get(font, 0)
                    count_style = styles.get(style, 0)
                    count_size = sizes.get(character.size, 0)
                    
                    
                    fonts[font] = count_font + 1
                    sizes[character.size] = count_size + 1
                    styles[style] = count_style + 1

                    
    # Most popular font and size in text
    font = sorted(fonts.items(), key=lambda x: x[1], reverse=True)[0][0] if fonts else 'Other'
    size = sorted(sizes.items(), key=lambda x: x[1], reverse=True)[0][0] if sizes else 0
    style = sorted(styles.items(), key=lambda x: x[1], reverse=True)[0][0] if styles else 'Other'
    return font, size, style


def calculate_len_score(len_value: float, mean_len: float, slopes: Tuple[float, float] = (2, 1.5)):
    def sigmoid(x, slope):
        return 1 / (1 + np.exp(-slope * x))

    slope = slopes[1] if len_value > mean_len else slopes[0]
    normalized_len = (len_value - mean_len) / mean_len
    
    return sigmoid(normalized_len, slope=slope)


def calculate_statistics(pages, verbose=False, verbose_params={}, min_length=50) -> dict:    
    attrs = ['width', 'x0', 'x1', 'font', 'size', 'style', 'length']
    elements = pd.DataFrame(columns=attrs)
    
    for pagenum, page in tqdm(enumerate(pages), total=verbose_params.get("n_pages")):         
        for i, element in enumerate(page._objs):
            if isinstance(element, LTTextContainer) and not element.is_empty():                    
                element.font, element.size, element.style = get_text_format(element)
                element.length = len(element.get_text())
                
                element_data = {attr: getattr(element, attr) for attr in attrs}
                elements.loc[len(elements.index)] = element_data
    
    font, size, style = elements.loc[:, ['font', 'size', 'style']].mode().iloc[0]
    hist_stats = ('hist', 'bin_edges')
    
    statistics = {
        'min_length': min_length,
        'style': style,
        'size': size,
        'font': font,
        'mean_length': elements.length.mean(),
        'width': dict(zip(hist_stats, np.histogram(elements.width))),
        'x0': dict(zip(hist_stats, np.histogram(elements.x0))),
        'x1': dict(zip(hist_stats, np.histogram(elements.x1)))
    }
    
    if verbose:
        print("Most common font:", font)
        print("Most common size:", size)
        print("Most common style:", style)
    
        fig, axs = plt.subplots(3, 2, figsize=(12, 8))
        axs = axs.flatten()
        bins = 20
        
        for i, attr in enumerate(['width', 'x0', 'x1', 'length']):
            ax = axs[i]
            ax.hist(elements[attr], bins=bins, alpha=0.7)
            ax.axvline(elements[attr].mean(), color='red', linestyle='dashed', linewidth=1)
            ax.set_title(attr.capitalize() + ' Distribution')
            ax.set_xlabel(attr.capitalize())
            ax.set_ylabel('Frequency')
            ax.grid(True)        
    
        if verbose and (slopes := (verbose_params.get("len_scores"))):
            ax = axs[4]
            length_scores = elements.length.apply(calculate_len_score, mean_len=elements.length.mean(), slopes=slopes)
        
            sns.scatterplot(x=elements.length, y=length_scores, ax=ax)
            ax.axvline(elements.length.mean(), color='g', linestyle='--', label='Mean Length')
            ax.set_xlabel('Length')
            ax.set_ylabel('Length Score')
            ax.set_title('Length Scores Distribution')
        
        plt.tight_layout()
        plt.show()
        
    return statistics


def get_attr_score(value, bin_edges, bin_scores):
    for i, bin_edge in enumerate(bin_edges):
        if bin_edge > value:
            bin_index = i - 1
            break
    else:
        bin_index = -1
    
    return bin_scores[bin_index] / bin_scores.max()


def get_page_num_score(page_num, total_pages, slope=50):
    x = page_num / total_pages
    return 1 / (1 + np.exp(-slope * (0.5 - abs(0.5 - x))))


def calculate_confidence(scores: dict, weights: dict):
    numerator = sum([score * weights[attr] for attr, score in scores.items()])
    denominator = sum(weights.values())
    
    return numerator / denominator


def get_semantic_scores(text: str, lang='eng'):
    result = {
        'regex': 0,
        'regex_msg': '',
        'verbs': 0
    }
    
    def validate_with_regex(text: str) -> Tuple[bool, str]:
        if re.match(r'(?:\d{1,2}\.\d{1,2})', text):
            return False, 'START_WITH_NUMBERS'
            
        if re.search(r'(?:. ){5,}', text, flags=re.DOTALL):
            return False, 'REPEATED_SEQUENCE'
        
        if (start_with_num_rows := re.findall(r'^\d\.?.*?$', text)) and (len(start_with_num_rows) > 2):
            return False, 'START_WITH_NUM_ROWS'
        return True, 'SUCCESS'
    
    # Validate with regex
    is_valid, validation_message = validate_with_regex(text)
    result['regex'] = 0.5 if is_valid else 0
    result['regex_msg'] = validation_message
    
    # Count verbs in text
    if lang == 'eng':
        verbs_tag = 'VB'
    elif lang in ('rus', 'ru'):
        lang = 'rus'
        verbs_tag = 'V'
    else:
        raise Exception('Supports only English and Russian languages')
        
    tokens = nltk.word_tokenize(text.lower())
    text = nltk.Text(tokens)
    tags = nltk.pos_tag(text, lang=lang)
    verbs = [tag for tag in tags if tag[1].startswith(verbs_tag)]
    
    result['verbs'] = len(verbs) / len(tags)
    
    # Linebreaks score
    linebreaks = text.count('\n')
    linebreaks_score = (1 - 0.2 * linebreaks) if linebreaks < 5 else 0
    result['linebreaks'] = linebreaks_score
    
    return result

def parse_pdf(filepath: str,
              conf_threshold: float,
              attr_weights: dict = None,
              statistics: dict=None,
              len_slopes=None,
              lang='eng',
              verbose: bool = False,
              debug: bool = False,
              debug_dir: str = None
             ):
    
    if attr_weights is None:
        attr_weights = copy.deepcopy(_DEFAULT_WEIGHTS)
    
    if len_slopes is None:
        len_slopes = copy.deepcopy(_DEFAULT_LEN_SLOPES)
    
    elements_to_include = []

    if debug:
        if not debug_dir:
            raise ValueError('"debug_dir" argument should be passed if debug=True')
            
        debug_filepath = debug_dir + filepath.split('/')[-1].removesuffix('.pdf') + '_CHECK.pdf'
        debug_pdf = canvas.Canvas(debug_filepath, pagesize=A4)
        debug_pdf.setTitle('parsedFile.pdf')
        
    file = open(filepath, 'rb')
    pages = extract_pages(file)
    parser = PDFParser(file)
    document = PDFDocument(parser)
    n_pages = resolve1(document.catalog['Pages'])['Count']
        
    if not statistics:
        print('Statistics dictionary not passed, calculate statistics...')
        verbose_params={
            "n_pages": n_pages,
            "len_scores": len_slopes
        }
        statistics = calculate_statistics(pages=extract_pages(file),
                                          verbose=verbose,
                                          verbose_params=verbose_params
                                         )
        print(f'Statistics calculated.\nResults:\n{statistics}')
        
    min_length = 50
    
    weights_0 = {name: attr_weights[name].pop('total') for name in attr_weights}
    
    for pagenum, page in tqdm(enumerate(pages), total=n_pages):         
        for i, element in enumerate(page._objs):
            if isinstance(element, LTTextContainer) and not element.is_empty():
                element.text = element.get_text()
                element.font, element.size, element.style = get_text_format(element)
                
                scores = {}
                pos_scores = {}
                semantical_scores = {}
                
                if len(element.text) < min_length:
                    confidence = 0
                else:
                    
                    pos_scores['length'] = calculate_len_score(len(element.text),
                                                           mean_len=statistics['mean_length'],
                                                           slopes=len_slopes)
                    pos_scores['pagenum'] = get_page_num_score(page_num=pagenum, total_pages=n_pages)
                    
                    for attr in ('x0', 'x1', 'width'):
                        pos_scores[attr] = get_attr_score(getattr(element, attr),
                                                      bin_edges=statistics[attr]['bin_edges'],
                                                      bin_scores=statistics[attr]['hist'])                    
                    scores['positional'] = calculate_confidence(pos_scores, weights=attr_weights['positional'])
                    
                    semantical_scores = get_semantic_scores(element.text)
                    semantical_scores['verbs'] *= pos_scores['length']
                    
                    regex_msg = semantical_scores.pop('regex_msg')
                    scores['semantical'] = calculate_confidence(semantical_scores, weights=attr_weights['semantical'])
                    
                    confidence = calculate_confidence(scores, weights_0)
                    
                    if confidence > conf_threshold:
                        elements_to_include.append((element.text, confidence))
                    
                color = 'grey' if confidence < conf_threshold else 'green'
                    
                if debug:
                    debug_pdf.setFillColor(color, alpha=0.2)
                    debug_pdf.rect(element.x0, element.y0, element.width, element.height, stroke=True, fill=True)
                    
                    # Текст элемента
                    text = debug_pdf.beginText(element.x0 + 1, element.y1 - element.size + 2)
                    text.setFont("Times-Roman", element.size)
                    text.setFillColor(color)
                    
                    for line in element.text.split('\n'):
                        text.textLine(line)
                    debug_pdf.drawText(text)
                    
                    # Текст справа от элемента с его параметрами
                    attrs_text = debug_pdf.beginText(element.x1, element.y1 - element.size + 2)
                    attrs_text.setFont("Courier", 7)
                    attrs_text.setFillColor(color)
                    attrs_text.textLine(f'Conf: {confidence}')
                    
                    for attr, value in scores.items():
                        attrs_text.textLine(f'{attr.capitalize()} Score: {value:.2f}')
                    
                    attrs_text.setFont("Courier", 5)
                    for attr, value in pos_scores.items():
                        attrs_text.textLine(f'{attr.capitalize()} Score: {value:.2f}')
                    
                    for attr, value in semantical_scores.items():
                        attrs_text.textLine(f'{attr.capitalize()} Score: {value:.2f}')
                        if attr == 'regex' and value < 0.5:
                            attrs_text.textLine(f'Regex message: {regex_msg}')
                    debug_pdf.drawText(attrs_text)
        if debug:                  
            debug_pdf.showPage()
    if debug:
        debug_pdf.save()
    file.close()
    return elements_to_include