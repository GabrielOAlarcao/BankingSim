import numpy as np
from exogeneous_factors import ExogenousFactors
from sklearn.linear_model import LogisticRegressionCV

class Util:
    id = 0

    @staticmethod
    def get_random_uniform(max_size):
        return np.random.uniform(0, max_size)

    @staticmethod
    def get_random_log_normal(mean, standard_deviation):
        return np.random.lognormal(mean, standard_deviation)

    @staticmethod
    def get_random_default_probability(firm_size, firm_type):
        if firm_type == 'HighRisk':
           return ExogenousFactors.alphaParamHighRisk / (firm_size ** (1/ExogenousFactors.betaParamHighRisk))
        else:
           return ExogenousFactors.alphaParamLowRisk / (firm_size ** (1/ExogenousFactors.betaParamLowRisk))
  
    @staticmethod
    def get_firm_size(a, n_firms, distribution='pareto'):
        np.random.seed(ExogenousFactors.firmSizeSeed)
        if distribution == "lognormal":
           initial_draw = (np.random.lognormal(0,0.5, n_firms))
           adjustment = abs(min(initial_draw) - ExogenousFactors.minFirmSize)
           firm_size = initial_draw + adjustment

        else: # Pareto (Default distribution)
           firm_size = (np.random.pareto(a, n_firms) + ExogenousFactors.minFirmSize) * ExogenousFactors.paretoDistributionMode
        return firm_size
      
    def get_power_law(alpha, x, beta):
        return alpha / (x ** (1/beta))

    @classmethod
    def get_unique_id(cls):
        cls.id += 1
        return cls.id

class FirmTypeModel:
    def __init__(self):
        self.beta0 = ExogenousFactors.beta0FirmTypeModel
        self.beta1 = ExogenousFactors.beta1FirmTypeModel
        self.beta2 = ExogenousFactors.beta2FirmTypeModel
        self.model = LogisticRegressionCV()

    def generate_data(self):
        np.random.seed(ExogenousFactors.modelSeed)
        x1 = (np.random.pareto(ExogenousFactors.alphaParetoDataGeneration,
                             size=100) + ExogenousFactors.minFirmSize) * ExogenousFactors.paretoDistributionMode
        # initial_draw = (np.random.lognormal(0,0.5, 100))
        # adjustment = abs(min(initial_draw) - ExogenousFactors.minFirmSize)
        # x1 = initial_draw + adjustment
        x2 = np.random.normal(0, 1, 100)
        z = self.beta0 + self.beta1 * x1 + self.beta2 * x2
        y = np.where(z > 0, 0, 1)
        X = np.column_stack((x1, x2))
        return y, X

    def train_model(self, X, y):
        self.model.fit(X, y)

    def generate_firm_risk_type(self, firm_size):
        y, X = self.generate_data()
        self.train_model(X, y)
        X_pred = self.model.predict_proba([[firm_size, 0]])[:,0]
        random_variable = np.random.random()
        
        if random_variable < X_pred:
           firm_type = 'LowRisk'
        else:
           firm_type = 'HighRisk'

        return firm_type 

    def predict_risk_type(self, firm_size):
        X_pred = np.array([[firm_size, 0]])
        risk_type = self.model.predict(X_pred)
        
        if risk_type == 0:
           pred_risk_type = 'LowRisk'
        else:
           pred_risk_type = 'HighRisk'
           
        return pred_risk_type 
