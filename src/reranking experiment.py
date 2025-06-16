# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 19:05:19 2025

@author: Natalia
"""

#Script que realiza el experimento de Rerankeo.

import networkx as nx
import numpy as np
import pandas as pd
import math
import random
import tools
from collections import OrderedDict
import first_interaction_index
import second_interaction_index
import link_prediction
import time


class lp_second_experiment():
    
    def __init__(self, graph, k_index, number_remove, batches_size, similarity_measure, method= 'score', weighted= False):
        """
        Método de inicialización del experimento de rerankeo. Este nos sirve para poner en común todas las características
        que comparten las diferentes creaciones de experimentos. Además de los valores insertados por parámetros se crearán ciertas
        variables necesarias como el número de rankings generados, una lista que almacena cada uno de los rankings, una lista que nos
        indica las aristas que han sido insertadas en el proceso, etc.
        
        Parameters
        -----------
        graph: Networkx graph.
            Grafo sobre el que vamos a realizar el experimento.
        k_index: int.
            Número entero que nos definirá el radio para aplicar nuestra medida de similitud.
        number_remove: int. 
            Número de aristas a quitar del grafo para realizar la comparación. 
        batches_size: int. 
            Entero el cual nos define el tamaño de los lotes de aristas que insertaremos paso a paso en el grafo.
        similarity_measure: String.
            Cadena de texto que nos indica la medida de similitud que emplearemos para realizar el experimento.
        method: String.
            Cadena que nos define el criterio seguido para insertar aristas en el grafo: orden en el ranking o aleatorio.
        weighted: Boolean value.
            Valor booleano que nos dice si el grafo es ponderado o no.     
        """
        self.graph= graph
        self.k_index= k_index
        self.weighted= weighted
        self.number_remove= number_remove
        self.batches_size= batches_size
        self.similarity_measure= similarity_measure
        self.method= method
        self.n_rankings= math.ceil(number_remove/batches_size)
        self.edges_to_remove= []
        self.all_possible_edges= {(u,v) for u in self.graph.nodes for v in range(1, u)}
               
    def execute_experiment(self):
        """
        Método que se encarga de ejecutar el experimento a lo largo de los diferentes instantes 
        de tiempo determinados por el número de aristas a eliminar y el tamaño de los 
        lotes en los que queremos insertarlas. El objetivo de esta función es almacenar
        los diferentes rankings que se dan tras la inserción de las aristas por lotes, para
        que éstos puedan ser analizados a posteriori a tavés de otras métricas. 

        Returns
        -------
        None.

        """
        inserted_edges= []
        rankings_list= [_ for _ in range(self.n_rankings)]
        graph_removed= self.graph.copy()
        self.edges_to_remove= random.sample(list(self.graph.edges), self.number_remove) #se sobreescribe
        graph_removed.remove_edges_from(self.edges_to_remove)
        self.edges_to_remove= [(edge[0], edge[1]) if edge[0]> edge[1] else (edge[1], edge[0]) for edge in self.edges_to_remove ]
        rankings_list[0]= self.compute_ranking(graph_removed)
        for i in range(self.n_rankings-1): #aquí iteramos sobre el número de inserciones
            edges_to_insert= self.compute_insertion(rankings_list[i]) #quiero que me devuelva un set
            inserted_edges.extend(edges_to_insert)
            graph_removed.add_edges_from(edges_to_insert)
            rankings_list[i+1]= self.compute_ranking(graph_removed)
        edges_to_insert= self.compute_insertion(rankings_list[i+1])
        inserted_edges.extend(edges_to_insert)
        first_ranking= list(rankings_list[0].keys())[: self.number_remove]
        precision1, precision2= self.precision_evaluation(self.edges_to_remove, first_ranking, inserted_edges)
        return (precision1, precision2)
        
        
    def compute_ranking(self, graph_removed):
        """
        Función que me computa el ranking del índice de interacción correspondiente
        a las aristas eliminadas del grafo en función de cuál sea la medida de similitud. 
        El objetivo será devolver un OrderedDict que almacene los pares (arista, índice) en
        orden ascendente (ya que los índices de inetracción son negativos) de los índices. 

        Parameters
        ----------
        graph_removed : netwotkx Graph or Digraph.
            Grafo de prueba que no contiene las aristas de estudio.
        k_index: int.
            Entero que nos indica la k-vecindad a tener en cuenta a la hora de realizar los rankings.
        weighted: boolean. 
            Valor booleano que nos indica si el grafo con el que estamos trabajando es ponderado o no.
        removed_edges : list.
            Lista que contiene tuplas las cuales representan las aristas eliminadas del
            grafo para el estudio de la predicción sobre ellas.
        similarity_measure : String.
            Cadena de texto que nos indica la medida de similitud que queremos utilizar.

        Returns
        -------
        ranking: OrderedDictionary.
            Diccionario ordenado de aristas según su índice de interacción. 
        """
        ranking= {}
        existent_links= {(u, v) if u> v else (v, u) for u, v in graph_removed.edges}
        non_existent_links= list(self.all_possible_edges - existent_links)
        if self.similarity_measure== 'SVII':
            tools.k_neighbors(graph_removed, self.k_index, self.weighted)
            interaction_index= first_interaction_index.shapley_II_extended(graph_removed)
            for edge in non_existent_links:
                ranking[edge]= interaction_index[edge[0]-1][edge[1]-1] if edge[0]> edge[1] else interaction_index[edge[1]-1][edge[0]-1] 
        elif self.similarity_measure== 'SVMII':
            tools.subconjuntos(graph_removed, self.k_index, self.weighted)
            interaction_index= first_interaction_index.tarkowski_shapley_II_extended(graph_removed)
            for edge in non_existent_links:
                ranking[edge]= interaction_index[edge[0]-1][edge[1]-1] if edge[0]> edge[1] else interaction_index[edge[1]-1][edge[0]-1] 
                #la comprobación que se hace aquí de que el primer vértice de la arista sea mayor al otro ya no haría falta 
        elif self.similarity_measure== 'CNN':
             tools.k_neighbors(graph_removed, self.k_index, self.weighted)
             for edge in non_existent_links:
                 ranking[edge]= first_interaction_index.common_neighbors_similarity(graph_removed, edge[0], edge[1])
        elif self.similarity_measure== 'AAI':
             tools.k_neighbors(graph_removed, self.k_index, self.weighted)
             for edge in non_existent_links:
                 ranking[edge]= first_interaction_index.adamic_adar_index(graph_removed, edge[0], edge[1])
        else:
            raise ValueError("La métrica de similitud insertada no es soportada elija entre SVII, SVMII, CNN, AAI")
        return  OrderedDict(sorted(ranking.items(), key= lambda x: x[1]))
    
    def precision_evaluation(self, edges_to_remove, first_ranking, inserted_edges):
        """
        Método de la clase que nos devuelve la precisión del primer ranking y la precisión de las aristas insertadas
        a través de la reevaluación de la clasificación.

        Returns
        -------
        p1 : float.
            Precisión de la primera clasificación dada.
        p2 : float.
            Precisión de las aristas insertadas a través de la reevaluación.

        """
        n= len(edges_to_remove)
        print("Imprimimos aristas borradas")
        print(edges_to_remove)
        print("Imprimimos primeras aristas recomendadas")
        print(first_ranking)
        print("Imprimos aristas insertadas al final")
        print(inserted_edges)
        precision_ranking, precision_reranking= 0, 0
        for edge in edges_to_remove:
            precision_ranking+=1 if edge in first_ranking else 0
            precision_reranking+=1 if edge in inserted_edges else 0
        p1= (precision_ranking/n)*1000
        p2= (precision_reranking/n)*1000
        return (p1, p2)
        

    def compute_insertion(self, ranking):
        """
        Función que elige batches_size aristas del ranking insertado siguiendo el método
        de selección pasado por parámetro.
    
        Parameters
        ----------
        ranking : OrderedDictionary.
            DESCRIPTION.
        batches_size : int.
            Entero que nos define el tamaño de los lotes a insertar en nuestra red.
        method : String.
            Cadena de texto que nos dice cuál es el método a seguir para elegir batches_size
            aristas a insertar en el graph_removed. Nos encontramos con las siguientes opciones:
                - random: Elije batches_size aristas al azar del ranking.
                - score: Elije las batches_size primeras aristas del ranking. 
                - neighborhood: Elije las batches_size aristas más cercanas. 
    
        Returns
        -------
        edges_to_insert: set.
            Conjunto de aristas a insertar en graph_removed
        """
        edges= []
        if self.method== 'random':
            edges= random.sample(list(ranking.values()), self.batches_size)
        elif self.method== 'score':
            edges= list(ranking.keys())[:self.batches_size]
        else:
            raise ValueError("El método de selección de aristas no es soportado por favor elija entre random o score.")
        return set(edges)
        
        
"""        
graph_zachary= link_prediction.read_data("zachary")
zachary= lp_second_experiment(graph_zachary, 3, 12, 3, 'SVII', 'score')
zachary.execute_experiment()
        
graph_football= link_prediction.read_data("football")
football= lp_second_experiment(graph_football, 3, 35, 1, 'SVMII', 'score')
football.execute_experiment()

graph_pequeño= link_prediction.read_data("pequeño")
pequeño= lp_second_experiment(graph_pequeño, 2, 3, 1, 'SVMII', 'score')
pequeño.execute_experiment()

graph_zachary= link_prediction.read_data("zachary")
zachary= lp_second_experiment(graph_zachary, 2, 9, 1, 'SVII', 'score')
zachary.execute_experiment()

"""


def study_reranking(k_index, percentage, batches_size, similarity_measure, method= 'score', weighted= False):
    """

    Parameters
    ----------
    k_index : TYPE
        DESCRIPTION.
    percentage : TYPE
        DESCRIPTION.
    batches_size : TYPE
        DESCRIPTION.
    similarity_measure : TYPE
        DESCRIPTION.
    method : TYPE, optional
        DESCRIPTION. The default is 'score'.
    weighted : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    results : TYPE
        DESCRIPTION.

    """
    beginning= time.perf_counter()
    graph_zachary= tools.read_data("zachary")
    graph_dolphins= tools.read_data("dolphins")
    graph_football= tools.read_data("football")
    graph_polbooks= tools.read_data("polbooks")
    graph_jazz= tools.read_data("jazz")
    zachary_experiment= lp_second_experiment(graph_zachary, k_index, round(percentage*len(graph_zachary.edges)), batches_size, similarity_measure, method, weighted)
    dolphins_experiment= lp_second_experiment(graph_dolphins, k_index, round(percentage*len(graph_dolphins.edges)), batches_size, similarity_measure, method, weighted)
    polbooks_experiment= lp_second_experiment(graph_polbooks, k_index, round(percentage*len(graph_polbooks.edges)), batches_size, similarity_measure, method, weighted)
    football_experiment= lp_second_experiment(graph_football, k_index, round(percentage*len(graph_football.edges)), batches_size, similarity_measure, method, weighted)
    jazz_experiment= lp_second_experiment(graph_jazz, k_index, round(percentage*len(graph_jazz.edges)), batches_size, similarity_measure, method, weighted)
    p1_zachary, p2_zachary, p1_dolphins, p2_dolphins, p1_polbooks, p2_polbooks, p1_football, p2_football, p1_jazz, p2_jazz= np.zeros(20), np.zeros(20), np.zeros(20), np.zeros(20), np.zeros(20), np.zeros(20), np.zeros(20), np.zeros(20), np.zeros(20), np.zeros(20)
    for i in range(20):
        p1_zachary[i], p2_zachary[i]= zachary_experiment.execute_experiment()
        p1_dolphins[i], p2_dolphins[i]= dolphins_experiment.execute_experiment()
        p1_polbooks[i], p2_polbooks[i]= polbooks_experiment.execute_experiment()
        p1_football[i], p2_football[i]= football_experiment.execute_experiment()
        p1_jazz[i], p2_jazz[i]= jazz_experiment.execute_experiment()
    mean_first_precision= [np.mean(p1_zachary), np.mean(p1_dolphins), np.mean(p1_polbooks), np.mean(p1_football), np.mean(p1_jazz)]
    mean_second_precision= [np.mean(p2_zachary), np.mean(p2_dolphins), np.mean(p2_polbooks), np.mean(p2_football), np.mean(p2_jazz)]
    results= pd.DataFrame({
        "Dataset": ['Zachary', 'Dolphins', 'Polbooks', 'Football', 'Jazz'], 
        "First precision": mean_first_precision, 
        "Second precision": mean_second_precision})
    end= time.perf_counter()
    execution_time= end-beginning
    print(f"The execution time was {execution_time:.5f} segundos")
    return results


def precision_evaluation(edges_to_remove, first_ranking, inserted_edges):
     """
     Método de la clase que nos devuelve la precisión del primer ranking y la precisión de las aristas insertadas
     a través de la reevaluación de la clasificación.

     Returns
     -------
     p1 : float.
         Precisión de la primera clasificación dada.
     p2 : float.
         Precisión de las aristas insertadas a través de la reevaluación.

     """
     n= len(edges_to_remove)
     precision_ranking, precision_reranking= 0, 0
     for edge in edges_to_remove:
         precision_ranking+=1 if edge in first_ranking else 0
         precision_reranking+=1 if edge in inserted_edges else 0
     p1= (precision_ranking/n)*1000
     p2= (precision_reranking/n)*1000
     return (p1, p2)








    
    
    

        
