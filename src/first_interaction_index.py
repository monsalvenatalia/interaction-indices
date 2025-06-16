# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:27:42 2025

@author: Natalia
"""

#En este fichero almacenamos las funciones que nos permiten el cálculo de las medidas de similitud:
#k-common neighbors, adamic-adar index, índice de interacción de k-grado de Shapley (SVII), índice de interacción
#de k-pasos amortiguado (SVMII). A toda medida de similitud le corresponden dos funciones de cálculo, aquella que se llama
#extended es la función que nos calcula a la vez las similitudes entre todos los vértices del grafo. Mientras que, la otra
#calcula la similitud entre los vértices insertados.

import tools 
import networkx as nx
from math import log
import numpy as np
import time


def common_neighbors_similarity(graph, s, t):
    """
    Función que calcula el número de vecinos comunes dentro de una k-vecindad para el par de nodos
    del grafo insertado. El resultado será su valor opuesto, es decir, multiplicándolo por menos uno,
    ya que a la hora de evaluar cuál se recomienda más, los índices de interacción calculados siguen ese criterio.
    
    Previamente a llamar a esta función ya se ha hecho el cálculo de las k-vecindades de los nodos del grafo.

    Parameters
    ----------
    graph : Graph from Netwotkx.
        Grafo o digrafo que representa la red de agentes, el cual debe tener almacenado para cada nodo un atributo
        que represente su k-localidad llamado k_neighbors. Además los nodos están representados a través de números 
        enteros. 
    s, t: int.
        Nodos cuyo índice de interacción queremos calcular. 

    Returns
    -------
    common_neighbors: int.
        Número de vecinos comunes en las k-vecindades del nodo s y t.
    """
    common_neighbors= -len(set(graph.nodes[s]['k_neighbors'])&set(graph.nodes[t]['k_neighbors']))
    return common_neighbors

def common_neighbors_extended(graph):
    """
    Función que cálcula la similitud en términos de número de vecinos comunes dentro de una vecindad de radio k. El conjunto
    de los vecinos dentro de la k-vecindad para cada uno de los nodos es computado previamente a través de la función k_neighbors 
    del script tools. 
    
    Parameters
    ----------
    graph : Graph from Networkx.
        Grafo o digrafo que representa la red de agentes, el cual debe tener almacenado para cada nodo un atributo que represente 
        su k-localidad llamado k_neighbors.

    Returns
    -------
    common_neighbors : squared numpy array.
        Matriz de numpy triangular inferior con ceros en la diagonal que almacena para cada 
        par de nodos el índice de interacción entre ellos, se accede a él tomando el nodo
        con índice más mayor como fila y el de menor índice como columna.

    """
    common_neighbors= np.zeros((len(graph.nodes), len(graph.nodes)), dtype= float)
    for s in sorted(graph.nodes, reverse= True):
        for t in range(1, s):
            common_neighbors[s-1][t-1]= -len(set(graph.nodes[s]['k_neighbors'])&set(graph.nodes[t]['k_neighbors']))
    return common_neighbors

def adamic_adar_index(graph, s , t):
    """
    Función que cálcula el índice de similitud de Adamic Adar entre dos nodos de un grafo. 
    
    Parameters
    ----------
    graph: Graph from Networkx.
        Grafo o digrafo que representa la red de agentes, el cual debe tener para cada nodo un atributo que represente su k-localidad 
        llamado k-neighbors.
    s, t: int.
        Nodos del grafo graph.
        
    Returns
    ----------
    adamic_adar_index: float.
        Índice de similitud de Adamic Adar entre los nodos s, t, opuesto en signo para que siga la misma lógica que los índices de interacción.
    """
    adamic_adar_index=0
    intersection= set(graph.nodes[s]['k_neighbors'])&set(graph.nodes[t]['k_neighbors'])
    for node in intersection:
        adamic_adar_index-= 1/(log(len(graph.nodes[node]['k_neighbors'])))
    return adamic_adar_index
 
def adamic_adar_extended(graph):
    """
    Función que cálcula el índice de similitud de Adamic Adar entre todo par de nodos del grafo, el índice de similitud, a pesar
    de que en realidad es positivo, se da su valor opuesto, para poder seguir la misma lógica en los experimentos que con los índices
    de interacción. 

    Parameters
    ----------
    graph : Graph from Networkx.
        Grafo de estudio.

    Returns
    -------
    adamic_adar : squared numpy array.
        Matriz de numpy triangular inferior con ceros en la diagonal que almacena para cada 
        par de nodos el índice de similitud AAI entre ellos.

    """
    adamic_adar= np.zeros((len(graph.nodes), len(graph.nodes)), dtype= float)
    for s in sorted(graph.nodes, reverse= True):
        for t in range(1, s):
            index, intersection= 0, set(graph.nodes[s]['k_neighbors'])&set(graph.nodes[t]['k_neighbors'])
            for node in intersection:
                index-= 1/(log(len(graph.nodes[node]['k_neighbors'])))
            adamic_adar[s-1][t-1]= index
    return adamic_adar
            

def shapley_interaction_index(graph, s, t):
    """
    Función que calcula el índice de interacción de Shapley para el par de nodos insertado como parámetros 
    de la función. Dado que para hacer los cálculos del índice de interacción entre dos nodos
    cualesquiera del grafo, es necesario contar con la información de qué nodos se encuentran en la k-vecindad
    de cada uno de ellos, previamente a llamar a esta función se ha hecho uso de la función auxiliar
    k-neighbors. 

    Parameters
    ----------
    graph : Graph from Netwotkx.
        Grafo o digrafo que representa la red de agentes, el cual debe tener almacenado para cada nodo un atributo
        que represente su k-localidad llamado k_neighbors. Además los nodos están representados a través de números 
        enteros. 
    s, t: int.
        Nodos cuyo índice de interacción queremos calcular. 

    Returns
    -------
    interaction_index : float.
        Número flotante negativo o cero que indica el índice de interacción de k-grado de 
        Shapley. Cuánto menor sea el valor del índice de interacción, más similares serán los nodos.

    """
    interaction_index= 0
    for u in set(graph.nodes[s]['k_neighbors'])&set(graph.nodes[t]['k_neighbors']):
        interaction_index-= 1/(len(graph.nodes[u]['k_neighbors'])-1)
    if s in graph.nodes[t]['k_neighbors']:
        if len(graph.nodes[t]['k_neighbors'])>1:
            interaction_index-= 1/(len(graph.nodes[t]['k_neighbors'])-1)
        if len(graph.nodes[s]['k_neighbors'])>1:
            interaction_index-= 1/(len(graph.nodes[s]['k_neighbors'])-1)
    return interaction_index

def shapley_II_extended(graph):
    """
    Función que calcula el índice de interacción de k-grado de Shapley para cada par de nodos posible
    en un grafo, devolviendo así una matriz triangular estrictamente inferior.

    Parameters
    ----------
    graph : Graph from Netwotkx.
        Grafo o digrafo que representa la red de agentes, el cual debe tener almacenado para cada nodo un atributo
        que represente su k-localidad llamado k_neighbors. Además los nodos están representados a través de números 
        enteros. 

    Returns
    -------
    interaction_index : squared numpy array.
        Matriz de numpy triangular inferior con ceros en la diagonal que almacena para cada 
        par de nodos el índice de interacción entre ellos, se accede a él tomando el nodo
        con índice más mayor como fila y el de menor índice como columna.
    """
    beginning= time.perf_counter()
    interaction_index= np.zeros((len(graph.nodes), len(graph.nodes)), dtype= float)
    for s in sorted(graph.nodes, reverse= True): #ya me los ordena aquí en función de las etiquetas de los nodos n,..., 1
        for t in range(1, s):
            index= 0
            for u in set(graph.nodes[s]['k_neighbors'])&set(graph.nodes[t]['k_neighbors']):
                    index-= 1/(len(graph.nodes[u]['k_neighbors'])-1)
            if s in graph.nodes[t]['k_neighbors']:
                if len(graph.nodes[s]['k_neighbors'])>1:
                    index-= 1/(len(graph.nodes[s]['k_neighbors'])-1)
                if len(graph.nodes[t]['k_neighbors'])>1:
                    index-= 1/(len(graph.nodes[t]['k_neighbors'])-1)
            interaction_index[s-1][t-1]= index
    end= time.perf_counter()
    execution_time= end-beginning
    print(f"The execution time was {execution_time:.5f} segundos")
    return interaction_index

def tarkowski_shapley_II(graph, s, t):
    """
    Función de cálculo del índice de interacción definido por Tarkowski para el semivalor de 
    Shapley. Este cálculo no tiene en consideración el tamaño de las posibles coaliciones, es decir, 
    el parámetro l, luego su cálculo va a ser mucho más simple. 
    
    En esta función aprovecharemos el cálculo hecho en la función subconjuntos del script de tools
    y, simplemente para calcular los nodos que se encuentran a distancia menor o menor o igual que otra 
    de un nodo, le restaremos al total de nodos las cantidades correspondientes calculadas en subconjuntos. 
    Puesto que sería ineficiente llamar a la función subconjuntos cada vez que se ejecute el índice de inte
    racción entre dos vértices, este será un paso previo a utilizar esta función.

    Parameters
    ----------
    graph : Graph from Netwotkx.
        Grafo o digrafo que representa la red de agentes, el cual debe tener almacenado para cada nodo un atributo
        que represente su k-localidad llamado k_neighbors. Además los nodos están representados a través de números 
        enteros. 
    s, t: int.
        Nodos cuyo índice de interacción queremos calcular.
        
    Returns
    -------
    interaction_index: float.
        Valor del índice de interacción de tarkowski entre los nodos s, t del grafo. 
    """

    n, nodes= len(graph.nodes), graph.nodes
    interaction_index= 0
    for u in nodes:
        minimo= min(nodes[u]["k_neighbors_distances"].get(s, -1), nodes[u]["k_neighbors_distances"].get(t, -1))
        maximo= max(nodes[u]["k_neighbors_distances"].get(s, -1), nodes[u]["k_neighbors_distances"].get(t, -1))
        higher_distances= {dist for dist in nodes[u]["k_neighbors_distances"].values() if dist> maximo} 
        #de la vecindad k de u, y entonces calcularemos el número de posibles coaliciones a esa distancia.
        if minimo!= -1: #niguno de los dos se encuentra fuera de la k-vecindad, ambos no pueden ser el mismo nodo luego d!=0 siempre
            nodos_le= n- nodes[u]["subsets"][maximo][0]
            positive_mc= 0 if nodos_le==1 else 1/((maximo**2)*(nodos_le -1))
            negative_mc= 0
            if higher_distances:
                for dist in higher_distances:
                    nodos_le_dist= n- nodes[u]["subsets"][dist][0]
                    nodos_lt_dist= n- nodes[u]["subsets"][dist][1]
                    negative_mc += 0 if (nodos_le_dist==1 or nodos_lt_dist==1) else (1/(dist**2))*(1/(nodos_lt_dist -1) -1/(nodos_le_dist -1)) #mirar si puede existir el caso de denominador negativo
            interaction_index+= (negative_mc - positive_mc)
    return interaction_index

def tarkowski_shapley_II_extended(graph):
    """
    Función de cálculo del índice de interacción definido por Tarkowski para el semivalor de 
    Shapley. Esta hace exactamente lo mismo que la anterior, pero en lugar de computar el índice
    de interacción para el par de nodos indicado, lo calcula para todos los nodos del grafo. 

    Parameters
    ----------
    graph : Graph from Netwotkx.
        Grafo o digrafo que representa la red de agentes, el cual debe tener almacenado para cada nodo un atributo
        que represente su k-localidad llamado k_neighbors. Además los nodos están representados a través de números 
        enteros. 
        
    Returns
    -------
    interaction_index: squared numpy array.
        Matriz de numpy triangular inferior con ceros en la diagonal que almacena para cada 
        par de nodos el índice de interacción entre ellos, se accede a él tomando el nodo
        con índice más mayor como fila y el de menor índice como columna.

    """
    beginning= time.perf_counter()
    n= len(graph.nodes)
    interaction_index= np.zeros((n, n), dtype= float)
    for u in graph.nodes:
        distances_items= list(graph.nodes[u]["k_neighbors_distances"].items())
        subsets= graph.nodes[u]["subsets"]
        higher_distance= distances_items[0][1]
        negative_mc= 0
        while len(distances_items)> 1:
            s, d= distances_items.pop(0)
            if d!= higher_distance:
                nodos_lt_hd= n - subsets[higher_distance][1]
                nodos_le_hd= n - subsets[higher_distance][0]
                negative_mc+= 0 if (nodos_lt_hd==1 or nodos_le_hd==1) else (1/(higher_distance**2))*(1/(nodos_lt_hd -1) -1/(nodos_le_hd -1))
                higher_distance= d
            for (t, dt) in distances_items:
                nodos_le= n- subsets[d][0]
                positive_mc= 0 if nodos_le==1 else 1/((d**2)*(nodos_le -1))
                if s> t:
                   interaction_index[s-1][t-1]+= negative_mc - positive_mc 
                else:
                    interaction_index[t-1][s-1]+= negative_mc - positive_mc 
    end= time.perf_counter()
    execution_time= end-beginning
    print(f"The execution time was {execution_time:.5f} segundos")
    return interaction_index






