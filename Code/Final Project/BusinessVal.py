
import numpy as np

# --- Business assumptions ---
LOAN_AMOUNT = 167_484.3
INTEREST_RATE = 0.17

PROFIT_GOOD_LOAN = LOAN_AMOUNT * INTEREST_RATE   # +16,748.43
LOSS_BAD_LOAN   = -LOAN_AMOUNT                   # -167,484.30


def business_value(cm, name=""):
    """
    cm: 2x2 confusion matrix with order:
        [[TN, FP],
         [FN, TP]]
    name: label for printing
    """
    tn, fp, fn, tp = cm.ravel()

    good_loans = tn   # predicted 0, actual 0
    bad_loans  = fn   # predicted 0, actual 1

    total_profit = good_loans * PROFIT_GOOD_LOAN + bad_loans * LOSS_BAD_LOAN
    total_cases  = cm.sum()
    ev_per_case  = total_profit / total_cases

    print(f"{name:30s} | TN={tn:4d} FP={fp:4d} FN={fn:4d} TP={tp:4d} | "
          f"Total profit = {total_profit:,.2f} | EV/applicant = {ev_per_case:,.2f}")


# --- Confusion matrices from your plots ---

# Decision Tree
cm_dt_rs0    = np.array([[4452, 221],
                         [ 834, 493]])

cm_dt_rs1033 = np.array([[4461, 212],
                         [ 900, 427]])

cm_dt_rs2025 = np.array([[4464, 209],
                         [ 872, 455]])

# Logistic Regression
cm_lr_rs0    = np.array([[4562, 111],
                         [1017, 310]])

cm_lr_rs1033 = np.array([[4534, 139],
                         [1011, 316]])

cm_lr_rs2025 = np.array([[4524, 149],
                         [1005, 322]])

# Naive Bayes
cm_nb_rs0    = np.array([[2429, 2244],
                         [ 343,  984]])

cm_nb_rs1033 = np.array([[3247, 1426],
                         [ 437,  890]])

cm_nb_rs2025 = np.array([[3425, 1248],
                         [ 477,  850]])


# --- Print business value for each ---

business_value(cm_dt_rs0,    "DT  RS=0")
business_value(cm_dt_rs1033, "DT  RS=1033")
business_value(cm_dt_rs2025, "DT  RS=2025")

business_value(cm_lr_rs0,    "LogReg RS=0")
business_value(cm_lr_rs1033, "LogReg RS=1033")
business_value(cm_lr_rs2025, "LogReg RS=2025")

business_value(cm_nb_rs0,    "NB  RS=0")
business_value(cm_nb_rs1033, "NB  RS=1033")
business_value(cm_nb_rs2025, "NB  RS=2025")
