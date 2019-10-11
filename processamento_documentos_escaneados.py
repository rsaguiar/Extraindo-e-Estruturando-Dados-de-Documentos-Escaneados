# -*- coding: utf-8 -*-
"""Processamento-documentos-escaneados

# Hands-On: Processando documentos escaneados

Por [Denys Farias](mailto:denys.lf@gmail.com) e [Rafael Aguiar](mailto:sextaa@gmail.com)

# Objetivo

1. Converter um PDF de Cart�o de Ponto em imagens;

2. Extrair texto de imagens (OCR);

3. Estruturar texto para obter batidas do cartão de ponto.

# Acompanhando o código

# https://www.codepile.net/pile/YwMNAx5Q

# Baixando o PDF de Exemplo

https://drive.google.com/open?id=1B0K9Ea21_XDux1eLkP4_mfh6GQDoxZ7h

# Preparando o ambiente
"""

# Commented out IPython magic to ensure Python compatibility.
# Definindo a pasta /content como atual e limpando o conteúdo dela
# %cd /content
# %rm -rf *
# %ls


# Instalando a biblioteca cliente do Google Cloud Vision (requer reinicialização)
!pip install --upgrade google-cloud-vision


# Instalando bibliotecas para manipulação de pdf
!sudo apt install poppler-utils
!pip install pdf2image


# Instalando bibliotecas para plotar anotações no Google Colab
!pip install mpld3
!pip install "git+https://github.com/javadba/mpld3@display_fix"

"""# 1. Converter PDF em imagens"""

from google.colab import files
from pdf2image import convert_from_path
import os

# Carregar PDF
uploaded_files = files.upload()
filenames = list(uploaded_files.keys())
pdf_filename = filenames[0]


# Criar pasta para as imagens, se não houver
images_folder = "/content/pdf_images"
if not os.path.exists(images_folder):
  try:
    os.mkdir(images_folder)
  except OSError:  
    print ("Creation of the directory %s failed" % images_folder)
  else:  
    print ("Successfully created the directory %s " % images_folder)
else:
  print ("Directory %s already exists" % images_folder)


# Converter PDF em imagens
pdf_pages = convert_from_path(pdf_filename, dpi=200, output_folder=images_folder, fmt='png')

"""# 2. Extrair texto das imagens (OCR) com o Google Cloud Vision API

# https://cloud.google.com/vision/

## Detalhando a estrutura do retorno JSON

> *fullTextAnnotation é uma resposta hierárquica estruturada do texto extraído da imagem. Ele é organizado como Pages (páginas) → Blocks (blocos) → Paragraphs (parágrafos) → Words (palavras) → Symbols (símbolos):*

> - *Page é um conjunto de blocos, além de metainformações sobre a página: tamanhos, resoluções (a X e a Y podem ser diferentes) etc.*

> - *Block representa um elemento "lógico" da página. Por exemplo, uma área coberta por texto, uma imagem ou um separador entre colunas. Os blocos de texto e tabela contêm as principais informações necessárias para extrair o texto.*

> - *Paragraph é uma unidade estrutural de texto que representa uma sequência ordenada de palavras. Por padrão, as palavras são separadas por quebras.*

> - *Word é a menor unidade do texto. Ela é representada como um conjunto de símbolos.*

> - *Symbol representa um caractere ou um sinal de pontuação.*

> *fullTextAnnotation também fornece URLs para imagens da Web que correspondem em parte ou totalmente à imagem na solicitação.*

Fonte: https://cloud.google.com/vision/docs/fulltext-annotations?hl=pt-br

## Como criar e gerenciar chaves de conta de serviço

https://cloud.google.com/iam/docs/creating-managing-service-account-keys?hl=pt-br#iam-service-account-keys-create-console

## Credencial de testes para baixar

https://drive.google.com/open?id=1wQ28kdUYp9gT-rAWj8jJ-dn4DLiV6Atp

## Importando e configurando as credenciais do Google Cloud
"""

import os

uploaded_credentials_filename = list(files.upload().keys())

if uploaded_credentials_filename:
  # Definir a variável de ambiente GOOGLE_APPLICATION_CREDENTIALS para o caminho do arquivo JSON 
  os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = uploaded_credentials_filename[0]

  print("Credentials ready!")

else:
  print("Credentials not ready.")

"""## Carregando imagem"""

import io
from os import listdir
from google.cloud import vision
from google.cloud.vision import types

# The name of the image file to annotate
image_path = f'{images_folder}/{listdir(images_folder)[0]}'

# Loads the image into memory
with io.open(image_path, 'rb') as image_file:
    image_content = image_file.read()

"""## Utilizando a API do Google Cloud"""

from google.cloud import vision
from google.cloud.vision import types


client = vision.ImageAnnotatorClient()
gc_image = types.Image(content=image_content)

# Performs label detection on the image file
gc_response = client.text_detection(image=gc_image)
texts = gc_response.text_annotations

print('Texts:')
for text in texts:
  print('\n"{}"'.format(text.description))
  for vertex in text.bounding_poly.vertices:
    vertices = (['({},{})'.format(vertex.x, vertex.y)])
    print('bounds: {}'.format(','.join(vertices)))

"""# 3. Estruturar texto em batidas

## Considerações sobre o Bounding Box

> The bounding box for the block. The vertices are in the order of top-left, top-right, bottom-right, bottom-left. When a rotation of the bounding box is detected the rotation is represented as around the top-left corner as defined when the text is read in the 'natural' orientation. For example:

> when the text is horizontal it might look like:
```
    0----1
    |    |
    3----2
  ```
  
> when it's rotated 180 degrees around the top-left corner it becomes:
```    
    2----3
    |    |
    1----0
```
    
> and the vertex order will still be (0, 1, 2, 3).

Fonte: https://cloud.google.com/vision/docs/reference/rest/v1/images/annotate

## Reunindo as palavras extraídas
"""

page = gc_response.full_text_annotation.pages[0]
all_words = [word for block in page.blocks for paragraph in block.paragraphs for word in paragraph.words]

all_words

"""## Definindo funções auxiliares para plotagem"""

import mpld3
from mpld3 import plugins

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.pyplot import figure

import matplotlib.patches as patches


# Para preparar área de plotagem
def prepare_image_data(image_data):
	im = np.array(image_data, dtype=np.uint8)

	# Create figure and axes
	fig,ax = plt.subplots(1)
	
  # Adjust the 
	width = 15
	height = width * image_data.size[1] / image_data.size[0]
	fig.set_size_inches(width, int(height), forward=True)
	
	plugins.connect(fig, plugins.MousePosition(fontsize=14))
	mpld3.enable_notebook()
  
	# Display the image
	ax.imshow(im)

	return ax


# Para definir uma paleta de cores e visualizar melhor as anotações
def get_tab10_color_from_index(color_index):
	COLORS_COUNT = 10
	cmap = plt.cm.tab10
	color = cmap(color_index % COLORS_COUNT)
	return color


# Para plotar polígonos
def plot_polygon(ax, vertices, color_index, to_fill = False):
	color = get_tab10_color_from_index(color_index)
	points_series = [[vertex.x,vertex.y] for vertex in vertices]

	polygon = patches.Polygon(points_series,linewidth=1,edgecolor=color,fill=color if to_fill else None)
	ax.add_patch(polygon)


# Para anotar texto sobre a plotagem
def annotate(text, vertex):
	plt.annotate(text, (vertex.x, vertex.y))


# Para renderizar a plotagem
plot = lambda: mpld3.display()


# Para capturar texto de palavra
def get_word_text(word):
  return ''.join([symbol.text for symbol in word.symbols])

"""## Plotando palavras encontradas"""

from PIL import Image

image_data = Image.open(image_path)

ax = prepare_image_data(image_data)
for	item_index, word in enumerate(all_words):
  plt.title(f'{image_path} words')
  plot_polygon(ax, word.bounding_box.vertices, color_index= item_index, to_fill= False)
  annotate(get_word_text(word), word.bounding_box.vertices[1])

plot()

"""## Definido código auxiliar para agrupar as palavras por linhas"""

import statistics
import itertools

# Ordenando palavras verticamente
ordered_words = sorted(all_words, key=lambda word: word.bounding_box.vertices[0].y)


# Para estimar alturar de palavras
def get_height(word):
  left_height = abs(word.bounding_box.vertices[0].y - word.bounding_box.vertices[3].y)
  right_height = abs(word.bounding_box.vertices[1].y - word.bounding_box.vertices[2].y)
  return (left_height + right_height) // 2


# Para decidir se palavra é "estável", ou seja, possui menos variações no bounding box.
min_stable_word_len = 3

def is_stable(word):
  return len(get_word_text(word)) >= min_stable_word_len


# Para estimar referência de altura de palavras
sample_size_for_height_reference_words = 5

def get_word_reference_height(words):
  # sample words for line height estimation
  stable_words = (word for word in words if is_stable(word))
  height_reference_words = list(itertools.islice(stable_words, sample_size_for_height_reference_words))

  if not height_reference_words:
    height_reference_words = [words[0]]
  
  # minimize variation on words heights
  word_reference_height = statistics.median([get_height(height_reference_word) for height_reference_word in height_reference_words])
  
  return word_reference_height

word_reference_height = get_word_reference_height(ordered_words)


# Para estimar variação vertical entre palavras
def get_accepted_y_variation(word_reference_height):
  return (word_reference_height + 2) // 3

accepted_y_variation = get_accepted_y_variation(word_reference_height)

"""## Agrupando palavras filtradas em linhas"""

# Ignorando palavras muito pequenas, que tendem a ser ruído
no_small_ordered_words = [word for word in ordered_words if get_height(word) >= word_reference_height / 2]

lines = {}
lines[0] = []

current_line_index = 0
current_line_y = no_small_ordered_words[0].bounding_box.vertices[0].y

for word in no_small_ordered_words:

  current_y = word.bounding_box.vertices[0].y
  is_new_line_by_y_variation = abs(current_y - current_line_y) > accepted_y_variation
  
  if is_new_line_by_y_variation:
    
    line_sorted_horizontally = sorted(lines[current_line_index], key=lambda word: word.bounding_box.vertices[0].x)
    lines[current_line_index] = line_sorted_horizontally 
    
    current_line_index += 1
    lines[current_line_index] = []

    # reference Y for a line should be estimated between neighboring stable words
    is_long_enough_to_update_line_Y_reference = is_stable(word)
    
    if is_long_enough_to_update_line_Y_reference:
      current_line_y = current_y

  lines[current_line_index] = lines[current_line_index] + [word]
  current_line_y = current_y


# Exibindo texto das linhas
lines

"""### Plotando as linhas"""

ax = prepare_image_data(image_data)
for	item_index, words in lines.items():
  for word in words:
    plt.title(f'{image_path} lines')
    plot_polygon(ax, word.bounding_box.vertices, color_index= item_index, to_fill= False)
    annotate(get_word_text(word), word.bounding_box.vertices[2])

plot()

"""## Preparando código para auxiliar na identificação de linhas com datas de batidas"""

import re
import datetime


# Para limpar caracteres iniciais indesejados
TOKENS_TO_STRIP = ['.', '-', '_', ':', '*', '\'', '"', '`', '´', '|', '~', '^', 'º', 'ª', '<', '>', '°', ';', '¨']

def ltrip_words(words, tokens_to_strip = TOKENS_TO_STRIP):
  result = list(words)
  to_examine_beginning = any(words)
  
  while to_examine_beginning:
    text_to_strip = get_word_text(result[0])
    stripped_text = text_to_strip.lstrip(''.join(TOKENS_TO_STRIP))
    
    if stripped_text:
      to_examine_beginning = False
      continue
		
    result = result[1:]
    if not result:
      break
	
  return result


# Para estimar vértices do bounding box de um conjunto de símbolos
def estimate_word_vertices(symbols):
	if not symbols:
		raise ValueError('Could not estimate from empty symbols.')
    
	vertex_0 = symbols[0].bounding_box.vertices[0]
	vertex_1 = symbols[-1].bounding_box.vertices[1]
	vertex_2 = symbols[-1].bounding_box.vertices[2]
	vertex_3 = symbols[0].bounding_box.vertices[3]

	return [vertex_0, vertex_1, vertex_2, vertex_3]


# Para extrair uma data de uma linha
leading_date_words_count = 6

def extract_date(line_words):
  
  max_leading_date_words_count = min(len(line_words), leading_date_words_count)
  
  leading_words = line_words[:max_leading_date_words_count]
  
  leading_text = ''.join([get_word_text(word) for word in leading_words])
  
  match = re.search(r'\d{2}/\d{2}/\d{4}', leading_text)

  extracted_date = None
  
  if match:
    extracted_pattern = match.group()
    
    try:
      date_obj = datetime.datetime.strptime(extracted_pattern, '%d/%m/%Y')
      extracted_date = extracted_pattern
    except:
      pass
    
  return extracted_date, leading_text

"""## Filtrando linhas iniciando com datas"""

lines_by_dates = []

for _, line_words in lines.items():
  
  extracted_date, leading_text = extract_date(line_words)

  if extracted_date:
    date_index = leading_text.index(str(extracted_date))
    symbols = [symbol for word in words for symbol in word.symbols]
    date_symbols = symbols[date_index : date_index + len(extracted_date)]
    
    date_line = {
      'text': extracted_date,
      'vertices': estimate_word_vertices(date_symbols),
      'symbols': date_symbols,
      'time_records': []
    }
    lines_by_dates = lines_by_dates + [(date_line, line_words)]

lines_by_dates

"""### Plotando as linhas com datas iniciais"""

ax = prepare_image_data(image_data)
for	item_index, line_by_date in enumerate(lines_by_dates):
  date = line_by_date[0]
  words = line_by_date[1]
  for word in words:
    plt.title(f'{image_path} date lines')
    plot_polygon(ax, word.bounding_box.vertices, color_index= item_index, to_fill= False)
    annotate(get_word_text(word), word.bounding_box.vertices[2])

plot()

"""## Preparando código para a identificar o que é data da batida, o que é horário da batida e o que é observação"""

# Para limpar caracters finais indesejados
def rtrip_words(words, tokens_to_strip = TOKENS_TO_STRIP):
  result = list(words)
  to_examine_ending = any(words)
  
  while to_examine_ending:
    text_to_strip = get_word_text(result[-1])
    stripped_text = text_to_strip.rstrip(''.join(TOKENS_TO_STRIP))
    
    if stripped_text:
      to_examine_ending = False
      continue
      
    result = result[:-1]
    if not result:
      break
      
  return result

"""## Limpando e indexando palavras das linhas iniciadas com datas"""

from sklearn.cluster import KMeans

# Limpando e indexando palavras das linhas iniciadas com datas
words_to_group = []
words_to_group_indexes = []

for line_by_date in lines_by_dates:
  date_line = line_by_date[0]
  words = line_by_date[1]

  words = rtrip_words(words)

  for index, word in enumerate(words):
    words_to_group += [word]
    words_to_group_indexes += [(date_line, index)]


# Encontrando os grupos de palavras com o K-Means
k_groups_count = 3 # Date records, Time records, Observations
k_means_random_state = 0 # For reproducibility

samples = [[word.bounding_box.vertices[0].x,1] for word in words_to_group]
labels = KMeans(n_clusters= k_groups_count, random_state= k_means_random_state).fit_predict(samples)


# Identificando os labels do grupo das observações
positions = [sample[0] for sample in samples]
max_position = max(positions)
max_position_index = positions.index(max_position)
observation_labels = [labels[max_position_index]]


# Identificando os labels do grupo da data inicial
min_position = min(positions)
min_position_index = positions.index(min_position)
leading_date_labels = [labels[min_position_index]]


# Identificando os índices dos grupos da data inicial e das batidas
time_records_indexes = [None if label in observation_labels or label in leading_date_labels else words_to_group_indexes[index] for index, label in enumerate(labels)]
time_records_indexes = list(filter(lambda index: index, time_records_indexes))

leading_date_indexes = [None if label not in leading_date_labels else words_to_group_indexes[index] for index, label in enumerate(labels)]
leading_date_indexes = list(filter(lambda index: index, leading_date_indexes))

"""## Filtrando as batidas e associando às datas iniciais"""

import re
import statistics


# filter words to consider in date_lines
for line_by_date in lines_by_dates:
  date_line = line_by_date[0]
  words = line_by_date[1]

  current_date_composite_time_records_indexes = filter(lambda index: index[0] == date_line, time_records_indexes)
  current_date_direct_time_records_indexes = [index[1] for index in current_date_composite_time_records_indexes]
  time_records_words = [words[index] for index in current_date_direct_time_records_indexes]
  # lines_by_dates[line_index] = (lines_by_dates[line_index][0], time_records_words)

  current_date_composite_leading_date_indexes = filter(lambda index: index[0] == date_line, leading_date_indexes)
  current_date_direct_leading_date_indexes = [index[1] for index in current_date_composite_leading_date_indexes]
  leading_date_words = [words[index] for index in current_date_direct_leading_date_indexes]

  leading_date_text = ''
  for word in leading_date_words:
    leading_date_text += get_word_text(word)

  if not leading_date_text:
    continue

  time_records_text = ''
  for word in time_records_words:
    time_records_text += get_word_text(word)

  time_records_symbols = [symbol for word in time_records_words for symbol in word.symbols]

  time_records = []
  for finding in re.finditer(r'\d{2}[\.:]\d{2}', time_records_text):
    ini_index = finding.start()
    fim_index = finding.end()
    finding_symbols = time_records_symbols[ini_index:fim_index]

    finding_text = finding.group(0).replace('.',':')
    finding_vertices = estimate_word_vertices(finding_symbols)

    time_records += [{
      'text': finding_text,
      'vertices': finding_vertices,
      'symbols': finding_symbols
    }]

  date_line['time_records'] = time_records

lines_by_dates

"""### Plotando as batidas"""

ax = prepare_image_data(image_data)
for	item_index, line_by_date in enumerate(lines_by_dates):
  date = line_by_date[0]
  time_records = date['time_records']
  for time_record in time_records:
    plt.title(f'{image_path} time records')
    plot_polygon(ax, time_record['vertices'], color_index= item_index, to_fill= False)
    annotate(time_record['text'], time_record['vertices'][2])

plot()