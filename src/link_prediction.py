# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 22:47:25 2025

@author: Natalia
"""

#Este fichero almacena el código empleado en el experimento de la predicción de vínculos. 
#Para la realización de este creamos una clase con todas las herramientas necesarias, y presentamos 
#una función que se encarga de computar el estudio con las cinco redes indicadas en la memoria. 

import tools
import first_interaction_index
from second_interaction_index import interaction_indices_type2
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random
import statistics
import numpy as np
import heapq
from collections import OrderedDict
import ipdb
import time


class lp_first_experiment():
    
    def __init__(self, graph, k_index, weighted, similarity_measure):
        """
        Este método de inicialización de un experimento almacena los datos necesarios para proceder
        a realizar el estudio del impacto del índice de interacción de Shapley en problemas de predicción 
        de vínculos. 
        
        Parameters
        ----------
        graph : Graph o Digraph de Networkx.
            Red de jugadores para la cual vamos a realizar el experimento.
        weighted : Boolean
            Valor booleano que nos indica si las aristas del grafo tienen peso o no.
        k_index: int
            Entero que nos indica el radio de las localidades de vecinos que queremos considerar.
        similarity_measure: String.
            Cadena de texto que me indica el índice de similitud que quiero utilizar para hacer el estudio
            correspondiente de predicción de vínculos. Dentro de los posibles índices de similitud tenemos:
                - GSVII: General Shapley Value Interaction Index.
                - GBVII: General Banzhaf Value Interaction Index.
                - SVII: Shapley Value Interaction Index (Szczepásnki)
                - GSVMII: General Shapley Value Modified Interaction Index.
                - GBVMII: General Banzhaf Value Modified Interaction Index.
                - SVMII: Shapley Value Modified Interaction Index (Tarkowski).
                - CNN: Common Neighbors.
                - AAI: Adamic Adar Index.

        Returns
        -------
        None.

        """
        self.graph= graph
        self.all_possible_edges= {(u,v) for u in self.graph.nodes for v in range(1, u)} #conjunto de aristas que tendría el grafo completo con los nodos del nuestro
        self.weighted= weighted
        self.k_index= k_index
        self.similarity_measure= similarity_measure
        
    def compute_similarities(self, graph_removed):
        """
        Método de la clase lp_first_experiment que computa el índice de interacción que queremos estudiar en el 
        experimento para cada par de nodos no emparejados en el grafo de entrenamiento pasado como parámetro de entrada. 
        Para ello hace un estudio previo de quiénes son los nodos que no poseen arista entre ellos y hace llamamiento a las
        precomputaciones necesarias para cada uno de los algoritmos de similitud. 

        Parameters
        ----------
        graph_removed : TYPE
            DESCRIPTION.

        Returns
        -------
        ranking: Ordered Dictionary.
            Diccionario ordenado que surje de haber reordenado los pares (arista, indice) en orden ascendente del índice.

        """
        existent_links= {(u, v) if u> v else (v, u) for u, v in graph_removed.edges}
        non_existent_links= list(self.all_possible_edges - existent_links)
        ranking= {}
        if self.similarity_measure== 'CNN':
            tools.k_neighbors(graph_removed, self.k_index, self.weighted)
            for edge in non_existent_links:
                ranking[edge]= first_interaction_index.common_neighbors_similarity(graph_removed, edge[0], edge[1])
        elif self.similarity_measure== 'AAI':
            tools.k_neighbors(graph_removed, self.k_index, self.weighted)
            for edge in non_existent_links:
                ranking[edge]= first_interaction_index.adamic_adar_index(graph_removed, edge[0], edge[1])
        elif self.similarity_measure== 'SVII':
            tools.k_neighbors(graph_removed, self.k_index, self.weighted)
            for edge in non_existent_links:
                ranking[edge]= first_interaction_index.shapley_interaction_index(graph_removed, edge[0], edge[1])
        elif self.similarity_measure== 'SVMII':
            tools.subconjuntos(graph_removed, self.k_index, self.weighted)
            for edge in non_existent_links:
                ranking[edge]= first_interaction_index.tarkowski_shapley_II(graph_removed, edge[0], edge[1])
        elif self.similarity_measure== 'GSVII':
            tools.k_neighbors(graph_removed, self.k_index, self.weighted) #primero se realizan los cálculos de distancias sobre el grafo
            element= interaction_indices_type2(graph= graph_removed, k_index= self.k_index, weighted= self.weighted, semivalue= "Shapley") #y después se crea una clase con el grafo como atributo
            for edge in non_existent_links:
                ranking[edge]= element.semivalue_interaction_index(edge[0], edge[1])
        elif self.similarity_measure== 'GBVII':
            tools.k_neighbors(graph_removed, self.k_index, self.weighted)
            element= interaction_indices_type2(graph= graph_removed, k_index= self.k_index, weighted= self.weighted, semivalue= "Banzhaf")
            for edge in non_existent_links:
                ranking[edge]= element.semivalue_interaction_index(edge[0], edge[1])
        elif self.similarity_measure== 'GSVMII':
            tools.subconjuntos(graph_removed, self.k_index, self.weighted)
            element= interaction_indices_type2(graph= graph_removed, k_index= self.k_index, weighted= self.weighted, semivalue= "Shapley")
            classement= element.tarkowski_semivalue_II()
            for edge in non_existent_links:
                ranking[edge]= classement[edge[0]-1][edge[1]-1]
        elif self.similarity_measure== 'GBVMII':
            tools.subconjuntos(graph_removed, self.k_index, self.weighted)
            element= interaction_indices_type2(graph= graph_removed, k_index= self.k_index, weighted= self.weighted, semivalue= "Banzhaf")
            classement= element.tarkowski_semivalue_II()
            for edge in non_existent_links:
                ranking[edge]= classement[edge[0]-1][edge[1]-1]
        else:
            raise ValueError("Medida de similitud no soportada elija alguna de las siguientes: CNN, SVII, SVMII, GSVII, GBVII, GSVMII, GBVMII")
        return OrderedDict(sorted(ranking.items(), key= lambda x: x[1]))
    
    def precision(self, graph_removed, num_missing_links, ranking):
        """
        Método que calcula la precisión total del índice de interacción de Shapley. 
        El método empleado es tomar el número de aristas eliminadas del grafo: missing_links y 
        tomar missing_links primeras aristas rankeadas según el índice de interacción. Comprobamos
        si estas aristas son efectivamente de las "faltantes" o del grupo de las "no existentes". 
        La métrica final será predichas_faltantes/missing_links.

        Parameters
        ----------
        graph_removed : Graph or Digraph from Networkx.
            Grafo de entrenamiento al que le hemos quitado un porcentaje de sus aristas.
        num_missing_links: int
            Nos aporta el número de aristas que han sido eliminadas del grafo para el entrenamiento.
        ranking: Ordered Dictionary.
            Diccionario ordenado de la librería collections que contiene pares de nodos sin conectar en el 
            grafo de entrenamiento, ordenados de manera ascendente en función de su índice de interacción. 

        Returns
        -------
        precision: int.
            precision*1000.

        """
        top_links= list(ranking.keys())[: num_missing_links] #[key for _, key in n_smallest]
        precision= 0
        for link in top_links:
            precision+= 1 if self.graph.has_edge(link[0], link[1]) else 0
        return (precision/num_missing_links)*1000
    
    def area_under_curve(self, graph_removed, edges_to_remove, ranking):
        """
        Método para el cálculo del área bajo la curva. 

        Parameters
        ----------
        graph_removed : Graph or Digraph from Networkx
            Grafo o digrafo al cual le hemos quitado un porcentaje de sus aristas. 
        edges_to_remove: list
            Lista de aristas eliminados del grafo original, es decir, missing_links.
        ranking: Ordered Dictionary. 
            Diccionario ordenado de la librería collections que contiene pares de nodos sin conectar en el 
            grafo de entrenamiento, ordenados de manera ascendente en función de su índice de interacción. 
            
        Returns
        -------
        AUC: float

        """
        right, wrong= 0,0 
        missing_links= [(u, v) if u> v else (v, u) for u, v in edges_to_remove]
        existent_links= {(u, v) if u> v else (v, u) for u, v in self.graph.edges}
        non_existent_links= list(self.all_possible_edges - existent_links)
        for m in missing_links:
            for l in non_existent_links:
                if ranking[m]< ranking[l]: #si el valor asignado a m es menor que el asignado a l, es más recomendado
                    right+=1
                elif ranking[m] == ranking[l]: #en caso contrario es igual o menos recomendado el missing_link que el non_existent
                    wrong+=1
        total= len(missing_links)*len(non_existent_links)
        return ((right+ wrong/2)/total)*1000
            
            
    def metricas_rendimiento(self, percentage= 0.4):
        """
        Método de la clase experimento que usaremos para evaluar las métricas 
        de precisión y area under the curve, insertando el porcentaje de aristas 
        que queremos eliminar de manera aleatoria para la evaluación del índice de 
        interacción del valor de Shapley.

        Parameters
        ----------
        percentage : float, optional
            Número perteneciente al intervalo [0,1] que indica el porcentaje de aristas a eliminar
            para el estudio. El valor por defecto es 0.4.

        Returns
        -------
        (precision, AUC): tuple
            Tupla con las dos métricas.

        """
        graph_removed= self.graph.copy()
        num_edges_remove= round(len(self.graph.edges)* percentage) #he cambiado la manera de coger el número de aristas a eliminar: antes int
        edges_to_remove= random.sample(list(self.graph.edges), num_edges_remove)
        graph_removed.remove_edges_from(edges_to_remove)
        ranking= self.compute_similarities(graph_removed)
        precision= self.precision(graph_removed, len(edges_to_remove), ranking)
        AUC= self.area_under_curve(graph_removed, edges_to_remove, ranking)
        return (precision, AUC)
            

def study(k_index= 2, percentage= 0.3, similarity_measure= 'SVII'):
    """
    Método que nos hace un estudio de las métricas de rendimiento del algoritmo de cálculo 
    del índice de interacción del valor de Shapley. Crea un experimento por dataset a 
    analizar y, tras insertar el valor del parámetro k nos devuelve el promedio de los valores de 
    las métricas obtenidos para 30 iteraciones.

    Parameters
    ----------
    k_index : int, optional
        El valor del radio de la localidad tomada. El valor por defecto es 2, aunque solemos cogerlos del conjunto
        {1, 2, 3}
    percentage: float.
        Número que indica el porcentaje de aristas del grafo que queremos eliminar para la creación de nuestro grafo de 
        entrenamiento.
    similarity_measure: String. 
        Medida de similitud empleada para computar la precisión y área bajo la curva en la tarea de predicción de vínculos.

    Returns
    -------
    results : Dataframe de pandas.
        Nos devuelve los promedios de las métricas para cada dataset.

    """
    beginning= time.perf_counter()
    graph_zachary= tools.read_data("zachary")
    graph_dolphins= tools.read_data("dolphins")
    graph_polbooks= tools.read_data("polbooks")
    graph_football= tools.read_data("football")
    graph_jazz= tools.read_data("jazz")
    zachary_experiment= lp_first_experiment(graph= graph_zachary, k_index= k_index, weighted= False, similarity_measure= similarity_measure)
    dolphins_experiment= lp_first_experiment(graph= graph_dolphins, k_index= k_index, weighted= False, similarity_measure= similarity_measure)
    polbooks_experiment= lp_first_experiment(graph= graph_polbooks, k_index= k_index, weighted= False, similarity_measure= similarity_measure)
    football_experiment= lp_first_experiment(graph= graph_football, k_index= k_index, weighted= False, similarity_measure= similarity_measure)
    jazz_experiment= lp_first_experiment(graph= graph_jazz, k_index= k_index, weighted= False, similarity_measure= similarity_measure)
    zachary_precision, zachary_auc, dolphins_precision, dolphins_auc, polbooks_precision, polbooks_auc, football_precision, football_auc, jazz_precision, jazz_auc= np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100)
    for i in range(100):
        zachary_precision[i], zachary_auc[i]= zachary_experiment.metricas_rendimiento(percentage)
        dolphins_precision[i], dolphins_auc[i]= dolphins_experiment.metricas_rendimiento(percentage)
        polbooks_precision[i], polbooks_auc[i]= polbooks_experiment.metricas_rendimiento(percentage)
        football_precision[i], football_auc[i]= football_experiment.metricas_rendimiento(percentage)
        jazz_precision[i], jazz_auc[i]= jazz_experiment.metricas_rendimiento(percentage)
    mean_precision= [np.mean(zachary_precision), np.mean(dolphins_precision), np.mean(polbooks_precision), np.mean(football_precision), np.mean(jazz_precision)]
    mean_auc= [np.mean(zachary_auc), np.mean(dolphins_auc), np.mean(polbooks_auc), np.mean(football_auc), np.mean(jazz_auc)]
    results= pd.DataFrame({
        "Dataset": ['Zachary', 'Dolphins', 'Polbooks', 'Football', 'Jazz'], 
        "Precision": mean_precision, 
        "AUC": mean_auc})
    end= time.perf_counter()
    execution_time= end-beginning
    print(f"The execution time was {execution_time:.5f} segundos")
    return results




    
    
    