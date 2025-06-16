
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 17:54:02 2024

@author: Natalia
"""

#Fichero que implementa las herramientas necesarias para el cálculo de las precomputaciones. 

import heapq
from collections import deque, OrderedDict
import networkx as nx
import math


def dijkstra(graph):
    """
    Función que para cada nodo del grafo o digrafo de la clase Networkx nos almacena
    una lista de distancias como nuevo atributo. De manera que, cada nodo tendrá a
    disposición la lista de sus distancias al resto de nodos del grafo graph.
    Usaremos este método siempre y cuando las aristas del grafo tengan pesos diferentes 
    a 1, ya que en caso contrario se usará la función breadth_first_search.

    Parameters
    ----------
    graph: Graph or Digraph from Networkx. 
        Grafo que almacena nuestra red de jugadores.

    Returns
    -------
    None.
    
    """
    for source in graph.nodes:
       distances= {node: float('inf') for node in graph.nodes}
       seen_nodes= {node: False for node in graph.nodes}
       distances[source]= 0
       priority_queue= [(0, source)]
       while priority_queue:
           current_distance, current_node= heapq.heappop(priority_queue)
           seen_nodes[current_node]= True
           for neighbor, attributes in graph[current_node].items():
               weight= attributes.get('weight', 1)
               if not seen_nodes[neighbor]:
                   if distances[neighbor]> distances[current_node] + weight:
                       distances[neighbor]= distances[current_node]+ weight
                       heapq.heappush(priority_queue, (distances[neighbor], neighbor))
       graph.nodes[source]['distances']= distances
       
       
def breadth_first_search(graph):
    """
    Función que a través del algoritmo de breadth first search busca los caminos mínimos
    entre cada uno de los nodos de un grafo y los almacena dentro de los atributos de un nodo 
    como un diccionario. La distancia establecida para aquellos nodos que no están conectados 
    entre sí será infinito. Este método lo usaremos siempre y cuando el grafo no posea aristas 
    con pesos, de manera que por defecto el peso de cada una de ellas será 1.

    Parameters
    ----------
    graph :  Graph or Digraph from Networkx
        Grafo que almacena nuestra red de jugadores.

    Returns
    -------
    None.

    """
    for source in graph.nodes:
        distances= {node: float('inf') for node in graph.nodes}
        distances[source]= 0
        queue= deque([source])
        while queue: 
            current_node= queue.popleft()
            for neighbor in graph.neighbors(current_node):
                if distances[neighbor]== float('inf'):
                    distances[neighbor]= distances[current_node]+1
                    queue.append(neighbor)
        graph.nodes[source]['distances']= distances
        
def compute_shapley(l, n):
    """
    Función auxiliar de la función semivalue_interaction_index que me calcula 
    el coeficiente por el cual tenemos que multiplicar la sinergia en el cálculo del 
    índice de interacción del semivalor de Shapley. Este me incluye la probabilidad 
    de que se forme una coalición de tamaño l, de todos los posibles tamaños de coaliciones del 
    grafo. 

    Parameters
    ----------
    l : int.
        Entero que me define el tamaño de una posible coalición en la red.
    n : int.
        Número de nodos que tiene el grafo.

    Returns
    -------
    float
        Factor por el cual debemos multiplicar la sinergia.

    """
    return 1/(math.comb(n-2, l)*(n-1))


#Funciones auxiliares correspondientes al cálculo de los índices de interacción definidos por Szczepásnki

def k_neighbors(graph, k, weighted):
    """
    Función que crea para cada nodo del grafo una lista con los nodos que se 
    encuentran en su k-vecindad y la almacena como atributo 'k-neighbors'.

    Parameters
    ----------
    graph : Graph or Digraph from Networkx.
        Grafo o digrafo que representa nuestra red de jugadores. Este grafo ha de cumplir
        que para cada nodo exista un diccionario como atributo que almacene la distancia que hay de dicho nodo
        a cada uno de sus vecinos.
    k : int
        Radio de la vecindad.

    Returns
    -------
    None.

    """
    if weighted:
        dijkstra(graph)
    else:
        breadth_first_search(graph)
    for node in graph.nodes:
        graph.nodes[node]['k_neighbors']= [key for key, value in graph.nodes[node]['distances'].items() if value<= k]
        graph.nodes[node]['k_neighbors'].remove(node)

#Funciones auxiliares correspondientes al cálculo de los índices de interacción definidos por Tarkowski

       
def k_neighbors_distances(graph, k, weighted):
    """
    Función que crea para cada nodo del grafo un diccionario el cual contiene como claves
    aquellos vértices del grafo que se encuentran en la k vecindad del nodo, y como valores
    almacenamos la distancia concreta a la cual se encuentran. Fijémonos en que esta distancia
    cumplirá ser menor o igual a k. Además este diccionario estará ordenado en orden descendente
    de los valores, de manera que el primer nodo del diccionario será aquel que se encuentra a mayor
    distancia y el último el más cercano.
    
    El diccionario creado será un atributo propio del nodo llamado "k_neighbors_distances". Es importante destacar
    que en este caso, a diferencia de en la función k-neighbors para el cálculo del índice de interacción
    introducido por Szczepásnki, no vamos a eliminar la distancia de un nodo a sí mismo, ya que nos será de utilidad 
    en operaciones posteriores. 
    
    Almacenaremos el par (nodo, 0) ya que el output de nuestra función va a ser clave para poder encontrar los conjuntos
    de nodos que se encuentran a distancias mayores o iguales a una distancia d existente dentro de la k-vecindad. En el 
    caso en que no almacenasemos este valor, el primer cálculo de nodos que se encuentran a distancia superior o igual a 
    la máxima existente dentro de la k-vecindad, sumaría un término de más. 

    Parameters
    ----------
    graph : Graph or Digraph from Networkx.
        Grafo o digrafo que representa nuestra red de jugadores. Este grafo ha de cumplir
        que para cada nodo exista un diccionario como atributo que almacene la distancia que hay de dicho nodo
        a cada uno de sus vecinos.
    k : int
        Radio de la vecindad.
    weighted: boolean.
        Este valor nos indica si nuestro grafo es ponderado o no para saber qué tipo de algoritmo de búsqueda
        de caminos mínimos poder aplicar. 

    Returns
    -------
    None.

    """
    if weighted:
        dijkstra(graph)
    else:
        breadth_first_search(graph)
    for node in graph.nodes:
        neighbors= {key: value for key, value in graph.nodes[node]['distances'].items() if value<= k}
        neighbors= OrderedDict(sorted(neighbors.items(), key= lambda x: x[1], reverse= True)) #usamos un diccionario ordenado
        graph.nodes[node]['k_neighbors_distances']= neighbors
        
def subconjuntos(graph, k, weighted):
    """
    Función que para cada nodo del grafo, y para distancia a la que se encuentra al menos uno de los 
    nodos de su k-vecindad, calcula dos cantidades:
        1. Cuántos nodos de la red se encuentran a una distancia mayor a dicha distancia. 
        2. Cuántos nodos de la red se encuentran a una distancia mayor o igual a dicha distancia. 
    Estas cantidades las almacenará en un diccionario, el cual para cada distancia, guarde como valores
    una lista que almacene las dos cantidades previas. Este diccionario será un atributo del nodo llamado
    subsets. 
    

    Parameters
    ----------
    graph : Grafo o Digrafo de Networkx.
        Grafo o digrafo que representa nuestra red de jugadores.
    k: int.
        Radio de la vecindad.
    weighted: boolean. 
        Este valor nos indica si nuestro grafo es ponderado o no para saber qué tipo de algoritmo de búsqueda 
        de caminos mínimos poder aplicar. 

    Returns
    -------
    None.

    """
    n= len(graph.nodes)
    k_neighbors_distances(graph, k, weighted)
    for node in graph.nodes:
        subsets_node= {}
        distances_node= list(graph.nodes[node]["k_neighbors_distances"].values()) #lista con las distancias ordenadas de mayor a menor dentro de la k-vecindad 
        #para el nodo node
        current_distance= distances_node[0]
        nodes_gt_d= n-len(distances_node)
        nodes_ge_d= nodes_gt_d
        for dist in distances_node:
            if dist!= current_distance:
                subsets_node[current_distance]= [nodes_gt_d, nodes_ge_d]
                nodes_gt_d= nodes_ge_d
                current_distance= dist
            nodes_ge_d+= 1
        subsets_node[current_distance]= [nodes_gt_d, nodes_ge_d]
        graph.nodes[node]["subsets"]= subsets_node
       
def read_data(dataset_name):
    """
    Función utilizada para leer el dataset que define cada una de las redes, 
    introduciendo como parámetro de entrada el nombre de la red que queremos 
    almacenar en el grafo de salida.

    Parameters
    ----------
    dataset_name : String
        Nombre del conjunto de datos que vamos a usar para construir nuestra red.

    Returns
    -------
    graph : Graph
        Grafo que contiene la información del dataset pasado por parámetro

    """
    file= f"./data/{dataset_name}/out_{dataset_name}.txt"
    graph= nx.read_edgelist(file, nodetype= int, comments= '%')
    return graph
       
       
       
       
       
        