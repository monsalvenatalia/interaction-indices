# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 23:06:06 2025

@author: Natalia
"""

import tools
from interaction_index import shapley_interaction_index
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random
import statistics
import numpy as np
import heapq
import ipdb
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


class clustering_experiment():
    
    def __init__(self, graph, k_index, weighted, similarity_measure= 'Modularity'):
        """
        Este método de inicialización de un experimento almacena los datos necesarios para proceder
        a realizar el estudio del impacto del índice de interacción de Shapley en problemas de 
        clusterización.

        Parameters
        ----------
        graph : Graph o Digraph de Networkx.
            Red de jugadores para la cual vamos a realizar el experimento.
        weighted : Boolean
            Valor booleano que nos indica si las aristas del grafo tienen peso o no.
        k_index: int
            Entero que nos indica el radio de las localidades de vecinos que queremos considerar.

        Returns
        -------
        None.

        """
        self.graph= graph
        self.weighted= weighted
        self.k_index= k_index
        self.similarity_measure= similarity_measure
        self.current_partition= {key: [value] for key, value in enumerate(graph.nodes)} #aunque para hacer el estudio de los dendrogramas vamos a tener que 
        #almacenar todas las particiones posibles. 
        #self.max_modularity= nx.modularity(self.graph, list(self.current_partition.values()))
        #self.best_partition= {key: [value] for key, value in enumerate(graph.nodes)}
        self.SVII= shapley_interaction_index(self.graph, self.k_index, self.weighted)
        
        
    def hierarchical_process(self):
        """
        Función que realizará la fusión continua de clusters e irá almacenando la modularidad máxima hasta 
        la fecha, junto con la partición que la genera.

        Parameters
        ----------
        None.
            
        Returns
        -------
        communities: dictionary
            The partition of the graph which gives us the best modularity result.
        modularity: float.
            The modularity metric of the best partition.
        """
        if len(self.current_partition)==1: #si la particióna actual tiene tamaño 1
            return (self.best_partition, self.max_modularity)
        else:
            return None
        
    def dendrogram(self):
        """
        Función que me devuelve el dendrograma del grafo producto de un proceso de clustering aglomerativo 
        con la función de similitud (distancia) del índice de interacción de Shapley, haciendo uso de la 
        técnica de complete linkage. 

        Returns
        -------
        None.

        """
        print(self.SVII)
        lowest_interaction_index= float('inf')
        for index in self.SVII.values():
            if index< lowest_interaction_index:
                lowest_interaction_index= index
        similarity_matrix= np.zeros((len(self.graph.nodes), len(self.graph.nodes)))
        for link in self.SVII.keys():
            column, row= link[0]-1, link[1]-1
            similarity_matrix[row][column]= self.SVII[link] + abs(lowest_interaction_index)
            similarity_matrix[column][row]= self.SVII[link] + abs(lowest_interaction_index)
        condensed_matrix= squareform(similarity_matrix)
        Z= linkage(condensed_matrix, method= 'complete')
        plt.figure(figsize=(20,6))
        dendrogram(Z, labels= range(1, 35), leaf_font_size= 10)
        plt.title("Dendrograma de Clustering Jerárquico (a partir de un grafo)")
        plt.show()
            
        
            
            
        
        
        
        
        