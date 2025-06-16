# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 11:27:01 2025

@author: Natalia
"""

import networkx as nx
import tools
import matplotlib.pyplot as plt
import math
import numpy as np
import heapq
from collections import OrderedDict

class interaction_indices_type2():
    
    def __init__(self, graph, k_index, weighted, semivalue):
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
            Cadena de texto que me indica el índice se similitud que quiero utilizar para hacer e

        Returns
        -------
        None.

        """
        self.graph= graph
        self.n= len(graph.nodes)
        self.weighted= weighted
        self.k_index= k_index
        self.semivalue = semivalue
        if semivalue == "Shapley":
            self.l_factors= [tools.compute_shapley(l, self.n) for l in range(0, self.n -1)]
        else:
            self.l_factors= None
            
    def tarkowski_semivalue_II(self):
        """
        Función que me calcula el índice de interacción del semivalor definido 
        en el artículo de Tarkowski. Para el cálculo de la función de distribución de 
        probabilidad discreta, puesto que se repite para cada par de nodos s, t su cálculo, 
        hemos decidido englobar el cálculo del factor correspondiente incluyendo el número
        combinatorio (|V|-2) sobre l, fuera de la función, y que este sea un parámetro de entrada 
        para el caso en el que se elija el semivalor de Shapley. Puesto que en el caso del 
        semivalor de Banzhaf, es cierto que la expresión se reduce a la división por la 
        constante 2*(n-2). 

            
        Raises
        ------
        ValueError
            En el caso de que se quiera calcular un índice de interacción procedente de 
            un semivalor distinto al de Shapley o el de Banzhaf, se levanta una excepción.

        Returns
        -------
        interaction_index: squared numpy array
            Matriz cuadrada triangular inferior con los índices de interacción entre cada par de nodos 
            diferentes.
        """
        #tools.subconjuntos(self.graph, self.k_index, self.weighted)
        n, nodes= self.n, self.graph.nodes
        interaction_index= np.zeros((n, n), dtype= float)
        if self.semivalue == "Shapley":
            for u in nodes:
                subsets= self.graph.nodes[u]["subsets"]
                for l in range(n-1):
                    distances_items= list(nodes[u]["k_neighbors_distances"].items())
                    higher_distance= distances_items[0][1]
                    negative_l= 0
                    while len(distances_items)>1:
                        s, d= distances_items.pop(0)
                        if d!= higher_distance:
                            nodos_ge_h= subsets[higher_distance][0]
                            nodos_gt_h= subsets[higher_distance][1]
                            negative_l+= (self.l_factors[l]/ higher_distance**2)*(math.comb(nodos_ge_h, l)- math.comb(nodos_gt_h, l))  
                            higher_distance= d
                        for (t, dt) in distances_items:
                            nodos_gt= subsets[d][1]
                            positive_l= (self.l_factors[l]/d**2)*math.comb(nodos_gt, l)
                            if s>t:
                                interaction_index[s-1][t-1]+= negative_l- positive_l
                            else:
                                interaction_index[t-1][s-1]+= negative_l- positive_l
        elif self.semivalue == "Banzhaf":
            for u in nodes:
                subsets= self.graph.nodes[u]["subsets"]
                for l in range(n-1):
                    distances_items= list(nodes[u]["k_neighbors_distances"].items())
                    higher_distance= distances_items[0][1]
                    negative_l= 0
                    while len(distances_items)>1:
                        s, d= distances_items.pop(0)
                        if d!= higher_distance:
                            nodos_ge_h= subsets[higher_distance][0]
                            nodos_gt_h= subsets[higher_distance][1]
                            negative_l+= (1/higher_distance**2)*(math.comb(nodos_ge_h, l)- math.comb(nodos_gt_h, l))  
                            higher_distance= d
                        for (t, dt) in distances_items:
                            nodos_gt= subsets[d][1]
                            positive_l= (1/d**2)*math.comb(nodos_gt, l)
                            if s>t:
                                interaction_index[s-1][t-1]+= (negative_l- positive_l)/(2**(n-2))
                            else:
                                interaction_index[t-1][s-1]+= (negative_l- positive_l)/(2**(n-2))
        else:
            raise ValueError("Semivalor no soportado elija bien Shapley o Banzhaf")
        return interaction_index
    
    def semivalue_interaction_index(self, s, t):
        """
        Función que me calcula el índice de interacción de k-grado de Shapley para cada par de 
        nodos del grafo. Esta función a raíz de un grafo y la especificación del semivalor del cual
        deriva nuestro índice de interacción, hará uso de la función auxiliar compute_shapley y 
        del factor constante de Banzhaf para calcular el factor el cual me incluye la función de la distribución de probabilidad 
        discreta correspondiente. 

        Parameters
        ----------
        s, t: int.
            Nodos cuyo índice de interacción queremos calcular.

        Raises
        ------
        ValueError
            En el caso de que se quiera calcular un índice de interacción procedente de 
            un semivalor distinto al de Shapley o el de Banzhaf, se levanta una excepción.

        Returns
        -------
        interaction_index : float.
            Valor del índice de interacción de Shapley reducido de Szczepásnki entre los nodos s, t.

        """
        #tools.k_neighbors(graph, k , weighted)
        n, nodes= self.n, self.graph.nodes
        interaction_index= 0
        if self.semivalue == "Shapley":
            for l in range(n -1):
                sinergy= 0
                for w in set(nodes[s]['k_neighbors'])&set(nodes[t]['k_neighbors']):
                    sinergy-= math.comb(n -1-len(nodes[w]['k_neighbors']), l)
                if s in nodes[t]['k_neighbors']:
                    sinergy-= math.comb(n-1-len(nodes[t]['k_neighbors']), l)
                    sinergy-= math.comb(n-1-len(nodes[s]['k_neighbors']), l)
                interaction_index+= self.l_factors[l]*sinergy
        elif self.semivalue == "Banzhaf":
            for l in range(n-1):
                sinergy= 0
                for w in set(nodes[s]['k_neighbors'])&set(nodes[t]['k_neighbors']):
                    sinergy-= math.comb(n-1-len(nodes[w]['k_neighbors']), l)
                if s in nodes[t]['k_neighbors']:
                    sinergy-= math.comb(n-1-len(nodes[t]['k_neighbors']), l)
                    sinergy-= math.comb(n-1-len(nodes[s]['k_neighbors']), l)
                interaction_index+= sinergy
            interaction_index/= (2**(n-2)) #el factor es igual para cualquier valor de l
        else:
            raise ValueError("Semivalor no soportado elija bien Shapley o Banzhaf")
        return interaction_index
                     

    
    
    
    
    
    
    
    
    
    
    
    
    
        
