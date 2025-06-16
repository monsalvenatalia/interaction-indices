# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 17:52:28 2024

@author: Natalia
"""


import networkx as nx
import tools
import matplotlib.pyplot as plt
import math
import heapq

#G= nx.karate_club_graph()
#distance_matrix= nx.floyd_warshall_numpy(G)


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
        tools.dijkstra(graph)
    else:
        tools.breadth_first_search(graph)
    for node in graph.nodes:
        graph.nodes[node]['k_neighbors']= [key for key, value in graph.nodes[node]['distances'].items() if value<= k]
        graph.nodes[node]['k_neighbors'].remove(node)
        #print(graph.nodes[node]['k_neighbors'])
        graph.nodes[node]['SVII']= {}
           
def common_neighbors_similarity(graph, k, weighted):
    """
    Función que calcula el número de vecinos comunes dentro de una k-vecindad para cada par de nodos
    del grafo. Devolveremos un diccionario de similaridad, el cual almacena esta medida para cada par
    de nodos dentro del grafo. Ordenaremos el valor de la intersección de las esferas a través de su valor 
    opuesto, es decir, multiplicándolo por menos uno, ya que a la hora de evaluar cuál se recomienda más, el índice de
    interacción de Shapley sigue ese criterio. 

    Parameters
    ----------
    graph : Graph from Netwotkx.
        Grafo o digrafo que representa la red de agentes, el cual debe tener almacenado para cada nodo un atributo
        que represente su k-localidad llamado k_neighbors. Además los nodos están representados a través de números 
        enteros. 
    k : int
        Radio de la localidad a emplear para el estudio de la similitud.

    Returns
    -------
    common_neighbors: dictionary.
        Diccionario donde las claves son pares de nodos (tuplas) y los valores son los índices 
        de similitud basados en la medida de vecinos comunes, dados con valores negativos, ordenados de manera ascendente. 
    """
    k_neighbors(graph, k, weighted)
    common_neighbors= {}
    for v in sorted(graph.nodes, reverse= True):
        for u in range(1, v): #recordemos que aquí estamos poniendo 1, porque las etiquetas de nuestros nodos son números enteros comenzando desde el 1
            common_neighbors[(v, u)]= -len(set(graph.nodes[v]['k_neighbors'])&set(graph.nodes[u]['k_neighbors']))
    common_neighbors= dict(sorted(common_neighbors.items(), key= lambda x: x[1]))
    return common_neighbors
 
def shapley_interaction_index(graph, k, weighted):
    """
    Función que calcula el índice de interacción de Shapley para cada par de nodos en un grafo, 
    combinando el cálculo de las k-vecindades y el índice de interacción. Además, para ahorrarnos tiempo 
    a la hora de evaluar el rendimiento del algoritmo, cada nodo almacenará como nuevo atributo un diccionario cuyas
    llaves serán los pares de nodos posibles en los que él se encuentra, y almacenaremos como valores los correspondientes
    índices de interacción.

    Parameters
    ----------
    graph : Graph from Netwotkx.
        Grafo o digrafo que representa la red de agentes, el cual debe tener almacenado para cada nodo un atributo
        que represente su k-localidad llamado k_neighbors. Además los nodos están representados a través de números 
        enteros. 
    k : int.
        Radio de la localidad a emplear para el estudio de la similitud.

    Returns
    -------
    interaction_index : dictionary.
        Diccionario donde las claves son pares de nodos (tuplas) y los valores son los índices 
        de interacción de Shapley entre ellos.

    """
    k_neighbors(graph, k, weighted)
    interaction_index= {}
    for v in sorted(graph.nodes, reverse= True): #ya me los ordena aquí en función de las etiquetas de los nodos n,..., 1
        for u in range(1, v):
            index= 0
            for n in set(graph.nodes[v]['k_neighbors'])&set(graph.nodes[u]['k_neighbors']):
                    index-= 1/(len(graph.nodes[n]['k_neighbors'])-1)
            if v in graph.nodes[u]['k_neighbors']:
                if len(graph.nodes[v]['k_neighbors'])>1:
                    index-= 1/(len(graph.nodes[v]['k_neighbors'])-1)
                if len(graph.nodes[u]['k_neighbors'])>1:
                    index-= 1/(len(graph.nodes[u]['k_neighbors'])-1)
            interaction_index[(v, u)]= index
            graph.nodes[v]['SVII'][(v, u)]= index #estas dos filas si vamos a trabajar con grafos no dirigidos sin
            graph.nodes[u]['SVII'][(u, v)]= index #hacer el promedio de las precisiones se pueden eliminar 
        graph.nodes[v]['SVII']= dict(sorted(graph.nodes[v]['SVII'].items(), key= lambda x: x[1]))
    interact_index= dict(sorted(interaction_index.items(), key= lambda x: x[1])) #recordar que aquí estamos tomando una nueva variable, no la misma de antes
    return interact_index

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

def semivalue_interaction_index(graph, k, weighted, semivalue):
    """
    Función que me calcula el k-steps semivalue índice de interacción para cada par de 
    nodos del grafo. Esta función a raíz de un grafo y la especificación del semivalor del cual
    deriva nuestro índice de interacción, hará uso de la función auxiliar compute_shapley y 
    del factor constante de Banzhaf para calcular el factor el cual me incluye la función de la distribución de probabilidad 
    discreta correspondiente. 

    Parameters
    ----------
    graph : networkx graph.
        Grafo de la librería networkx del cual queremos extraer información.
    k : int.
        Radio de la vecindad considerada para calcular el índice.
    weighted : boolean.
        Parámetro que nos dice si el grafo es ponderado o no.
    semivalue : String.
        Cadena que nos dice si queremos usar el semivalor de Shapley/Banzhaf.

    Raises
    ------
    ValueError
        En el caso de que se quiera calcular un índice de interacción procedente de 
        un semivalor distinto al de Shapley o el de Banzhaf, se levanta una excepción.

    Returns
    -------
    interaction_index : dictionary.
        Diccionario que almacena para cada par de nodos su respectivo índice de interacción.

    """
    n= len(graph.nodes)
    k_neighbors(graph, k , weighted)
    interaction_index= {}
    if semivalue == "Shapley":
        l_factor= [compute_shapley(l, n) for l in range(0, n-1)] #calculamos los posibles factores, si lo pasamos a una operación que sea par por par puede ser ineficiente tener el cálculo dentro
        for v in sorted(graph.nodes, reverse= True): 
            for u in range(1, v):
                index= 0
                for l in range(n-1):
                    sinergy= 0
                    for w in set(graph.nodes[v]['k_neighbors'])&set(graph.nodes[u]['k_neighbors']):
                            sinergy-= math.comb(n-1-len(graph.nodes[w]['k_neighbors']), l)
                    if v in graph.nodes[u]['k_neighbors']:
                        sinergy-= math.comb(n-1-len(graph.nodes[u]['k_neighbors']), l)
                        sinergy-= math.comb(n-1-len(graph.nodes[v]['k_neighbors']), l)
                    index+= l_factor[l]*sinergy
                interaction_index[(v, u)]= index
    elif semivalue == "Banzhaf":
        for v in sorted(graph.nodes, reverse= True): 
            for u in range(1, v):
                index= 0
                for l in range(n-1):
                    sinergy= 0
                    for w in set(graph.nodes[v]['k_neighbors'])&set(graph.nodes[u]['k_neighbors']):
                            sinergy-= math.comb(n-1-len(graph.nodes[w]['k_neighbors']), l)
                    if v in graph.nodes[u]['k_neighbors']:
                        sinergy-= math.comb(n-1-len(graph.nodes[u]['k_neighbors']), l)
                        sinergy-= math.comb(n-1-len(graph.nodes[v]['k_neighbors']), l)
                    index+= sinergy
                interaction_index[(v, u)]= index/(2**(n-2)) #el factor es igual para cualquier valor de l
    else:
        raise ValueError("Semivalor no soportado elija bien Shapley o Banzhaf")
    return dict(sorted(interaction_index.items(), key= lambda x: x[1]))
    
        
#el índice de interacción de shapley no tiene sentido para grafos dirigidos  
"""  
def shapley_interaction_index_directed(digraph, k, weighted):
    
    Función que dados un digrafo G y un entero k, que define el radio de nuestra localidad considerada, 
    nos devuelve un diccionario cuyas llaves son tuplas que definen un par de nodos y los valores almacenados 
    son los índices de interacción entre dichos nodos. Volvemos a almacenar los valores de los índices de interacción 
    de cada par en los que el nodo esté incluido en forma de diccionario llamado SVII.

    Parameters
    ----------
    graph : Digraph from Netwotkx.
        Grafo o digrafo que representa la red de agentes.
    k : int
        Radio de la localidad a emplear para el estudio de la similitud.

    Returns
    -------
    interaction_index: dictionary.
        Diccionario que almacena para cada par de nodos de la red el valor de su índice de interacción de Shapley.

    k_neighbors(digraph, k, weighted)
    interaction_index= {}
    for v in sorted(digraph.nodes, reverse= True):
        for u in range(1, v):
            index= 0
            for n in set(digraph.nodes[v]['k_neighbors'])&set(digraph.nodes[u]['k_neighbors']):
                if len(digraph.nodes[n]['k_neighbors'])>1:
                    index-= 1/(len(digraph.nodes[n]['k_neighbors'])-1)
            if v in digraph.nodes[u]['k_neighbors']:
                if len(digraph.nodes[u]['k_neighbors'])>1:
                    index-= 1/(len(digraph.nodes[u]['k_neighbors'])-1)
            if u in digraph.nodes[v]['k_neighbors']:
                if len(digraph.nodes[v]['k_neighbors'])>1:
                    index-= 1/(len(digraph.nodes[v]['k_neighbors'])-1)
            interaction_index[(v, u)]= index
            digraph.nodes[v]['SVII'][(v, u)]= index
            digraph.nodes[u]['SVII'][(u, v)]= index
        digraph.nodes[v]['SVII']= dict(sorted(digraph.nodes[v]['SVII'].items(), key= lambda x: x[1]))
    interaction_index= dict(sorted(interaction_index.items(), key= lambda x: x[1]))
    return interaction_index
"""
    

"""
def tarkowski_shapley_II_inicial(graph, s, t):
---------------
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
    k : int.
        Radio de la localidad a emplear para el estudio de la similitud.
    weighted: boolean.
        Valor booleano que nos indica si el grafo es ponderado o no.
    s, t: int.
        Nodos cuyo índice de interacción queremos calcular.
        
    Returns
    -------
    interaction_index: float.
        Valor del índice de interacción de tarkowski.

-------------
    n, nodes= len(graph.nodes), graph.nodes
    interaction_index= 0
    for u in nodes:
        d= max(nodes[u]["k_neighbors_distances"].get(s, 0), nodes[u]["k_neighbors_distances"].get(t, 0))
        higher_distances= set([dist for dist in nodes[u]["k_neighbors_distances"].values() if dist> d]) 
        #de la vecindad k de u, y entonces calcularemos el número de posibles coaliciones a esa distancia.
        if d!= 0:
            nodos_le= n- nodes[u]["subsets"][d][0]
            positive_mc= 1/((d**2)*(nodos_le -1))
            negative_mc= 0
            if higher_distances:
                for dist in higher_distances:
                    nodos_le_dist= n- nodes[u]["subsets"][dist][0]
                    nodos_lt_dist= n- nodes[u]["subsets"][dist][1]
                    negative_mc += (1/(dist**2))*(1/(nodos_lt_dist -1) -1/(nodos_le_dist -1)) #mirar si puede existir el caso de denominador negativo
            interaction_index+= (negative_mc - positive_mc)
    return interaction_index
"""
    
def print_intersection(graph): #función de comprobación de calculo de las intersecciones
    lista= {}
    for v in sorted(graph.nodes, reverse= True):
        for u in range(1, v):
            lista[(v,u)]= {"intersección": (set(graph.nodes[v]['k_neighbors'])&set(graph.nodes[u]['k_neighbors'])), "v_neighborhood": set(graph.nodes[v]['k_neighbors']), "u_neighborhood": set(graph.nodes[u]['k_neighbors'])}
    return lista
    
    
    
    