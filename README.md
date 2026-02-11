**Allocation de Portefeuille Robuste
De l’Optimisation Max Sharpe à une Stratégie Validée Hors Échantillon**

**Présentation**
Ce projet a pour objectif de construire, tester et stabiliser une stratégie d’allocation de portefeuille initialement fondée sur une optimisation 10 ans de type Maximum Sharpe Ratio.
L’approche consiste à comparer une allocation théoriquement optimale in-sample à des versions corrigées et validées hors échantillon afin d’identifier une structure réellement exploitable dans un cadre d’investissement long terme.

**Objectifs**
Construire une allocation 10Y Max Sharpe à partir des rendements historiques.
Évaluer sa robustesse via des tests hors échantillon.
Identifier les sources de sur-optimisation.
Mettre en place des méthodes de stabilisation statistiques.
Extraire une allocation robuste utilisable en gestion réelle.

**Méthodologie**
_1. Optimisation Max Sharpe naïve_
Estimation des rendements annualisés par moyenne historique.
Estimation de la matrice de covariance empirique.
Optimisation sous contraintes (poids maximum par actif).
Résultat :
Performance élevée en in-sample (~22 %), mais effondrement hors échantillon (~0 %).
Conclusion :
La stratégie est fortement dépendante de la période d’estimation et souffre de sur-ajustement.
_2. Stabilisation par Shrinkage_
Shrinkage des rendements vers un prior de marché.
Estimation robuste de la covariance via Ledoit–Wolf.
Optimisation Max Sharpe sous contraintes.
Résultat hors échantillon :
Rendement annualisé ≈ 10 %
Volatilité ≈ 19 %
Drawdown maîtrisé
Conclusion :
La stratégie devient statistiquement plus stable et économiquement plus crédible.
_3. Approche Black–Litterman_
Construction d’un prior d’équilibre de marché.
Intégration optionnelle de vues relatives.
Estimation robuste de la covariance.
Résultat :
Rendement plus élevé
Volatilité significativement plus importante
Profil de risque agressif
Conclusion :
Modèle performant mais adapté à un investisseur tolérant une forte variabilité.
_4. Validation Walk-Forward_
Fenêtre d’apprentissage roulante de 5 ans.
Rebalancement trimestriel.
Intégration des coûts de transaction.
Ce protocole simule les conditions réelles d’implémentation.

Résultats Hors Échantillon
Stratégie	CAGR	Volatilité	Max Drawdown	Robustesse
Max Sharpe naïf	~0 %	17 %	-32 %	Faible
Shrinkage	~10 %	19 %	-29 %	Solide
Black–Litterman	~26 %	48 %	-38 %	Agressif
L’optimisation naïve ne survit pas à la validation hors échantillon.
La version shrinkée fournit une structure robuste et exploitable.

Allocation Robuste Moyenne (Shrink – Hors Échantillon)
Actif	Poids moyen
MEDCL.PA	19.1 %
EMEIS.PA	17.8 %
VIG	13.4 %
IEF	12.3 %
STM_EUR	11.4 %
URTH	10.9 %
AELIS.PA	6.5 %
AF.PA	3.8 %
ALCRB.PA	3.2 %
KER.PA	1.5 %
Cette allocation est :
Diversifiée sectoriellement.
Moins concentrée que la version naïve.
Validée hors échantillon.
Compatible avec un horizon d’investissement long terme.
Tests de Robustesse Implémentés
Walk-forward hors échantillon.
Bootstrap Monte Carlo (horizon 10 ans).
Analyse de stabilité des poids sous perturbation.
Shrinkage des rendements.
Shrinkage de la covariance (Ledoit–Wolf).
Approche Black–Litterman.
Limites
Les résultats sont fondés sur des données historiques et ne garantissent pas les performances futures.
Le projet vise à construire un processus d’allocation discipliné, non à prédire les rendements futurs.

Conclusion
L’optimisation théorique seule ne suffit pas à construire une stratégie d’investissement robuste.
La validation hors échantillon et la stabilisation statistique sont indispensables pour transformer un modèle performant en un cadre d’allocation crédible et exploitable.
