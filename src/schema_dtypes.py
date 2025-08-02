# Columnas con tipos mixtos detectadas
mixed_type_columns = [
    "Sexo",  # categórica
    "Renta", "CUPO_L2", "CUPO_MX", "Fac_T12", "FacCN_T12", "FacCI_T12", "FacAI_T12", "Col_T12",
    "ColL1TE_T12", "ColMx_T12", "PagoInt_T12", "EeccNac_T12", "EeccInt_T12", "UsoL1_T12", "UsoLI_T12",
    "IndRev_T12", "Fac_T11", "FacCN_T11", "FacCI_T11", "FacAI_T11", "Col_T11", "ColL1TE_T11", "ColMx_T11",
    "PagoInt_T11", "EeccNac_T11", "EeccInt_T11", "UsoL1_T11", "UsoLI_T11", "IndRev_T11", "Fac_T10",
    "FacCN_T10", "FacCI_T10", "FacAI_T10", "Col_T10", "ColL1TE_T10", "ColMx_T10", "PagoInt_T10",
    "EeccNac_T10", "EeccInt_T10", "UsoL1_T10", "UsoLI_T10", "IndRev_T10", "Col_T09", "ColL1TE_T09",
    "ColMx_T09", "PagoInt_T09", "EeccNac_T09", "EeccInt_T09", "UsoL1_T09", "UsoLI_T09", "IndRev_T09",
    "Col_T08", "ColL1TE_T08", "ColMx_T08", "PagoInt_T08", "EeccNac_T08", "EeccInt_T08", "UsoL1_T08",
    "UsoLI_T08", "IndRev_T08", "Col_T07", "ColL1TE_T07", "ColMx_T07", "PagoInt_T07", "EeccNac_T07",
    "EeccInt_T07", "UsoL1_T07", "UsoLI_T07", "IndRev_T07", "Fac_T06", "FacCN_T06", "FacCI_T06",
    "FacAI_T06", "Col_T06", "ColL1TE_T06", "ColMx_T06", "PagoInt_T06", "EeccNac_T06", "EeccInt_T06",
    "UsoL1_T06", "UsoLI_T06", "IndRev_T06", "Col_T05", "ColL1TE_T05", "ColMx_T05", "PagoInt_T05",
    "EeccNac_T05", "EeccInt_T05", "UsoL1_T05", "UsoLI_T05", "IndRev_T05", "Col_T04", "ColL1TE_T04",
    "ColMx_T04", "PagoInt_T04", "EeccNac_T04", "EeccInt_T04", "UsoL1_T04", "UsoLI_T04", "IndRev_T04",
    "Fac_T03", "FacCN_T03", "FacCI_T03", "FacAI_T03", "Col_T03", "ColL1TE_T03", "ColMx_T03", "PagoInt_T03",
    "EeccNac_T03", "EeccInt_T03", "UsoL1_T03", "UsoLI_T03", "IndRev_T03", "Col_T02", "ColL1TE_T02",
    "ColMx_T02", "PagoInt_T02", "EeccNac_T02", "EeccInt_T02", "UsoL1_T02", "UsoLI_T02", "IndRev_T02"
]

# Separar las numéricas que realmente deberían ser floats
numeric_mixed_columns = [col for col in mixed_type_columns if col != "Sexo"]

# Diccionario de tipos forzados
forced_dtypes = {col: "float64" for col in numeric_mixed_columns}