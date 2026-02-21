from enum import Enum


class SimulationType(Enum):
    HighSpread = 1
    LowSpread = 2
    ClearingHouse = 3
    ClearingHouseLowSpread = 4
    Basel = 5
    BaselBenchmark = 6
    DepositInsurance = 7
    DepositInsuranceBenchmark = 8


class BankSizeDistribution(Enum):
    Vanilla = 1
    LogNormal = 2


class InterbankPriority(Enum):
    Random = 1
    RiskSorted = 2


class ExogenousFactors:
    # Model
    numberBanks = 50
    depositInterestRate = 0.07
    interbankInterestRate = 0.01  # Unused
    liquidAssetsInterestRate = 0.08
    illiquidAssetDiscountRate = 0.05
    interbankLendingMarketAvailable = True
    banksMaySellNonLiquidAssetsAtDiscountPrices = True
    banksHaveLimitedLiability = False

    # Banks
    bankSizeDistribution = BankSizeDistribution.Vanilla
    numberDepositorsPerBank = 3
    numberCorporateClientsPerBank = 2
    numberHighRiskCorporateClientsPerBank = 2
    areBanksZeroIntelligenceAgents = False
    knowsFirmType = False

    # Central Bank
    centralBankLendingInterestRate = depositInterestRate + 0.06
    offersDiscountWindowLending = True
    minimumCapitalAdequacyRatio = 0.08
    isCentralBankZeroIntelligenceAgent = True
    isCapitalRequirementActive = True
    isTooBigToFailPolicyActive = False
    isDepositInsuranceAvailable = False

    # Clearing House
    isClearingGuaranteeAvailable = False
    interbankPriority = InterbankPriority.Random

    # Depositors
    areDepositorsZeroIntelligenceAgents = True
    areBankRunsPossible = True
    amountWithdrawn = 1.0
    probabilityofWithdrawal = 0.15

    # Firms / Corporate Clients
    standardCorporateClients = True
    heterogeneousRiskDistribution = True
    demandMultiplier = 1.5
    standardCorporateClientLoanSize = demandMultiplier*(1 / (numberCorporateClientsPerBank+numberHighRiskCorporateClientsPerBank)) # 1*2  # 10
    standardCorporateClientElasticity = (.7 / (numberCorporateClientsPerBank+numberHighRiskCorporateClientsPerBank))  # (1/1.20)*2
    standardCorporateClientDefaultRate = 0.045
    alphaParamLowRisk = 0.45
    alphaParamHighRisk = 0.4
    alphaParamCreditDemand = 0.8
    alphaParamBank = 0.6
    betaParamLowRisk = 0.9
    betaParamHighRisk = 1.4
    betaParamBank = 0.6
    betaParamCreditDemand = 0.6
    alphaParetoDataGeneration = 5
    minFirmSize = 1
    beta0FirmTypeModel = -1.1
    beta1FirmTypeModel =  sizeRiskCorrelation = 0.3
    beta2FirmTypeModel = -0.65
    paretoDistributionMode = 3

    highRiskCorporateClientDefaultRate = 0.1
    highRiskCorporateClientLossGivenDefault = 1
    highRiskCorporateClientLoanInterestRate = 0.065
    highRiskCorporateClientLoanSize = 1 / (numberCorporateClientsPerBank+numberHighRiskCorporateClientsPerBank)  # 1*2  # 10
    highRiskCorporateClientElasticity = .7 / (numberCorporateClientsPerBank+numberHighRiskCorporateClientsPerBank)

    standardCorporateClientLossGivenDefault = 1
    standardCorporateClientLoanInterestRate = 0.08
    wholesaleCorporateClientDefaultRate = 0.04
    wholesaleCorporateClientLoanInterestRate = 0.06
    wholesaleCorporateClientLossGivenDefault = 1
    retailCorporateClientDefaultRate = 0.06
    # retailCorporateClientLoanInterestRate = 0.08
    # retailCorporateClientLossGivenDefault = 0.75

    # Risk Weights
    CashRiskWeight = 0
    CorporateLoanRiskWeight = 2
    HighRiskCorporateLoanRiskWeight = 5
    InterbankLoanRiskWeight = 1
    retailCorporateLoanRiskWeight = 0.75
    wholesaleCorporateLoanRiskWeight = 1

    # Learning
    DefaultEWADampingFactor = 1
    LambdaFactor = 1
    
    # Distribution
    firmSizeSeed = 20
    modelSeed = 42
