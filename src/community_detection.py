# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 12:20:26 2025

@author: Natalia
"""

#Este script realiza el procedimiento de detección de comunidades, para ello contiene una 
#clase para el almacenamiento de Clusters y otra para la realización del experimento. De manera adicional, 
#creamos funciones para el cálculo del coverage y para ejecutar los algoritmos Greedy y Louvain sobre
#los conjuntos de datos. 

import numpy as np
from time import time
import math
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import sys
import pandas as pd
import networkx as nx
import tools
import first_interaction_index
from tools import read_data
import community as community_louvain
from collections import defaultdict


class Cluster:
    ClusterCount = 0
    
    def __init__(self, nodes= []):
        """
        Método de inicialización de un elemento de la clase Cluster. Una vez generamos una nueva instancia de esta 
        clase se actualiza el número de clusters que existen, a través del atributo de clase ClusterCount. De esta manera, 
        Cluster Count nos define el número de clusters que se han creado de la clase, y hemos de tener en cuenta que los 
        identificadores de los clusters irán de 1 hasta ClusterCount. 

	    Parameters
	    ----------
	    nodes : list.
	        Lista que contiene los identificadores de los nodos que se encuentran en este cluster. Su valor por 
            defecto es [].

	    """
        Cluster.ClusterCount+=1
        self.id= Cluster.ClusterCount
        self.nodes= nodes

    def add_node(self, node_id):
        """
        Método de la clase que añade un nuevo nodo al cluster. 
    
        Pameters
        -------
        node_id : int.
        Identificador del nodo que se quiere añadir a este nuevo cluster.
        """
        self.nodes.append(node_id)


class clustering_experiment():
    
    def __init__(self, graph, k_index, weighted, similarity_measure):
        """
        Método de inicialización del experimento. Este inicializa los atributos que van a ser necesarios para el desarrollo del experimento. 
        Veamos cuáles son:
            - graph: El grafo en el que queremos encontrar una separación por comunidades.
            - k_index: El radio de la vecindad que tendremos en cuenta para calcular las similitudes entre nodos. 
            - weighted: Booleano que nos dirá si es ponderado o no.
            - similarity_measure: Nos indica la métrica con la que queremos calcular el índice de interacción entre nodos. 
            - modularity: será un array de numpy que almacenará la modularidad de cada configuración.
            - configurations: array de numpy que almacenará en cada fila las etiquetas que le corresponden a cada uno de los nodos en la configuración actual
            - linkage: será la matriz de fusiones de clusters que se pasará como parámetro de entrada a dendrograma de scipy.
            - clusters: diccionario que almacena todos los clusters generados en el proceso, a cada identificador de cluster le da el objeto de la clase Cluster
            - similarity_matrix: matrix original de índices de interacción entre los nodos del grafo, es simétrica con 0s en la diagonal.
            - similarity_df: dataframe que va almacenando los índices de interacción entre las comunidades actuales. 

        Parameters
        ----------
        graph : NetworkX graph.
        k_index : int.
        weighted : boolean.
        similarity_measure : String.

        Returns
        -------
        None.

        """
        Cluster.ClusterCount= 0
        self.graph= graph
        self.k_index= k_index
        self.weighted= weighted
        self.similarity_measure= similarity_measure
        self.modularity= np.zeros(len(graph.nodes), dtype= float)
        self.configurations= np.zeros((len(graph.nodes), len(graph.nodes)), dtype= int)
        self.linkage= np.zeros((len(graph.nodes)-1, 4), dtype= float)
        self.clusters= {}
        for node in range(1, len(graph.nodes)+1):
            cluster= Cluster([node])
            self.clusters[cluster.id]= cluster
        if similarity_measure== 'SVII':
            self.similarity_matrix= np.zeros((len(graph.nodes), len(graph.nodes)), dtype= np.float64)
            tools.k_neighbors(graph, k_index, weighted)
            M= first_interaction_index.shapley_II_extended(graph)
        elif similarity_measure== 'SVMII':
            self.similarity_matrix= np.zeros((len(graph.nodes), len(graph.nodes)), dtype= np.float64)
            tools.subconjuntos(graph, k_index, weighted)
            M= first_interaction_index.tarkowski_shapley_II_extended(graph)
        elif similarity_measure== 'CNN':
            self.similarity_matrix= np.zeros((len(graph.nodes), len(graph.nodes)), dtype= np.float64)
            tools.k_neighbors(graph, k_index, weighted)
            M= first_interaction_index.common_neighbors_extended(graph)
        elif similarity_measure== 'AAI':
            self.similarity_matrix= np.zeros((len(graph.nodes), len(graph.nodes)), dtype= np.float64)
            tools.k_neighbors(graph, k_index, weighted)
            M= first_interaction_index.adamic_adar_extended(graph)
        else:
            raise ValueError("La medida de similitud introducida no es reconocida, porfavor seleccione una entre: SVII, SVMII, CNN")
        self.similarity_matrix = M + M.T
        self.similarity_df= pd.DataFrame(self.similarity_matrix.copy())
        self.similarity_df.rename(index= lambda x: x+1, columns= lambda x: x+1, inplace= True)
        
        
    def merging_clusters(self):
        """
        Método de la clase que nos identifica cuáles son las siguientes dos comunidades
        a fusionar.

        Returns
        -------
        clusterA, clusterB : int
            Identificadores de las comunidades que queremos fusionar. 
        minimum_II : float.
            Índice de interacción entre las comunidades.

        """
        
        minimum_II= math.inf
        clusterA, clusterB= None, None
        for row in range(0, self.similarity_df.shape[0]-1):
            for column in range(row+1, self.similarity_df.shape[1]):
                if self.similarity_df.iloc[row, column]< minimum_II:
                    minimum_II= self.similarity_df.iloc[row, column]
                    clusterA, clusterB= row, column
        return clusterA, clusterB, minimum_II
    
    def merge_communities(self, clusterA_id, clusterB_id):
        """
        Método de la clase cluster experiment que realiza el proceso de fusión de clusters. Para ello, 
        lo que hace es: crear una nueva instancia de la clase clusters y ajustar la matriz de similitudes 
        para reflejar correctamente las nuevas relaciones entre el nuevo cluster y el resto.

        Parameters
        ----------
        clusterA_id : Objeto inmutable.
            Identificador o índice/nombre de columna del primero de los clusters que queremos unificar.
        clusterB_id : Objeto inmutable.
            Identificador o índice/nombre de columna del segundo de los clusters que queremos unificar.

        Returns
        -------
        new_cluster : Cluster() element.
            Nuevo cluster creado a raíz de la fusión de los otros dos.
        """
        new_cluster= Cluster(self.clusters[clusterA_id].nodes + self.clusters[clusterB_id].nodes)
        self.clusters[new_cluster.id]= new_cluster
        self.similarity_df[clusterA_id]= self.similarity_df[clusterA_id].combine(self.similarity_df[clusterB_id], max)
        self.similarity_df.loc[clusterA_id]= self.similarity_df.loc[clusterA_id].combine(self.similarity_df.loc[clusterB_id], max)
        self.similarity_df.drop(index= clusterB_id, inplace= True)
        self.similarity_df.drop(columns= clusterB_id, inplace= True)
        self.similarity_df.rename(index= {clusterA_id: new_cluster.id}, inplace= True)
        self.similarity_df.rename(columns= {clusterA_id: new_cluster.id}, inplace= True)
        print(self.similarity_df)
        return new_cluster
    
    def compute_modularity(self, configuration): #no hace falta pasarle el grafo porque ya lo tiene al alcance
        """
        Método de la clase que calcula la modularidad de una configuración dada. Este método recibe una configuración por clusters 
        a través de un numpy array y devuelve la modularidad que le corresponde haciendo uso del método modularity de la librería 
        networkx.
        
        Parameters
        ----------
        configuration: np.array.
            Array de numpy de una dimensión que contiene para cada nodo el identificador del cluster al que pertenece en la 
            configuración actual.
            
        Returns
        ---------
        La modularidad de la configuración. 
        """
        labels= set(configuration)
        clusters= []
        for label in labels:
            community= set()
            for i in range(len(configuration)):
                if configuration[i]== label:
                    community.add(i+1)
            clusters.append(community)
        return nx.algorithms.community.quality.modularity(self.graph, clusters)
        
    def execute_experiment(self):
        """
        Función de la clase que se encarga de ejecutar el experimento. 

        Returns
        -------
        max_modularity : float.
            El valor de la modularidad máxima conseguida a través del proceso de clustering aglomerativo.
        max_configuration : numpy array de una dimensión.
            Array que contiene para cada posición, la cual define un nodo en orden correspondiente (1, 2, .., n) el cluster asignado
            en la configuración con mayor modularidad. 
        """
        
        self.configurations[0, :]= np.arange(1, len(self.graph.nodes)+1)
        self.modularity[0]= self.compute_modularity(self.configurations[0, :])
        for i in range(1, self.graph.number_of_nodes()):
            clusterA, clusterB, minimum_II= self.merging_clusters()
            clusterA_id, clusterB_id= self.similarity_df.index[clusterA], self.similarity_df.index[clusterB]
            new_cluster= self.merge_communities(clusterA_id, clusterB_id)
            distancia= 4 if minimum_II==0 else -1/minimum_II
            self.linkage[i-1]= np.array([int(clusterA_id)-1, int(clusterB_id)-1, distancia, len(new_cluster.nodes)]) #actualizamos la matriz de linkage
            previous= self.configurations[i-1]
            new_configuration= np.where((previous== clusterA_id) | (previous== clusterB_id), new_cluster.id, previous)
            self.configurations[i]= new_configuration
            self.modularity[i]= self.compute_modularity(new_configuration)
        max_modularity= np.max(self.modularity)
        max_configuration= self.configurations[np.argmax(self.modularity)]
        max_coverage= compute_coverage(self.graph, max_configuration)
        plt.figure(figsize= (8,5))
        etiquetas= [str(i) for i in range(1, len(self.graph.nodes)+1)]
        dendrogram(self.linkage, orientation= 'right', labels= etiquetas)
        plt.show()
        return max_modularity, max_configuration, max_coverage
  
           
def compute_coverage(graph, configuration):
    """
    Método que calcula el coverage de una clusterización cualquiera de un grafo.

    Parameters
    ----------
    graph: Networkx Graph.
        Grafo de la librería Networkx.
    configuration : numpy array.
        Array de numpy que contiene en cada posición la comunidad a la que pertenece el nodo con 
        identificador la posición en el array +1.

    Returns
    -------
    float
        Medida de coverage de la división en comunidades del grafo actual.

    """
    
    node_community= {}
    for i in range(len(configuration)):
        node_community[i+1]= configuration[i]
    internal_edges= sum(1 for u, v in graph.edges() if node_community[u]== node_community[v])
    return internal_edges/(graph.number_of_edges())
   
    
def greedy_louvain_modularity():
    graphs= [read_data("zachary"), read_data("dolphins"), read_data("polbooks"), read_data("football")]
    mod_greedy, number_greedy, mod_louvain, number_louvain= [], [], [], []
    for i in range(4):
        communities_greedy = nx.community.greedy_modularity_communities(graphs[i])
        communities_greedy= [list(c) for c in communities_greedy]
        partition= community_louvain.best_partition(graphs[i])
        communities_louvain = defaultdict(set)
        for node, com in partition.items():
            communities_louvain[com].add(node)
        communities_louvain= list(communities_louvain.values())
        mod_greedy.append(nx.community.modularity(graphs[i], communities_greedy))
        number_greedy.append(len(communities_greedy))
        mod_louvain.append(nx.community.modularity(graphs[i], communities_louvain))
        number_louvain.append(len(communities_louvain))
    results= pd.DataFrame({
        "Dataset": ['Zachary', 'Dolphins', 'Polbooks', 'Football'], 
        "Greedy MOD": mod_greedy, 
        "Greedy nº communities": number_greedy,
        "Louvain MOD": mod_louvain, 
        "Louvain nº communities": number_louvain
        })
    return results
       
























