# Credit Risk Model

## Credit Scoring Business Understanding

### 1. How does Basel II influence model development?

Basel II requires financial institutions to measure, monitor, and manage credit risk using reliable and transparent methods. Because lending decisions affect regulatory capital requirements, models must be interpretable, well-documented, and auditable. Clear documentation allows regulators and business stakeholders to understand how risk predictions are produced and ensures compliance with regulatory standards.

### 2. Why is a proxy variable necessary?

The dataset does not contain a direct loan default label. Therefore, a proxy target variable must be created using customer behavioral patterns. In this project, customer engagement metrics such as Recency, Frequency, and Monetary value can be used to identify potentially high-risk customers.

However, proxy variables introduce business risks because they do not represent actual defaults. The model may learn customer inactivity rather than true repayment behavior, potentially reducing prediction reliability.

### 3. Trade-offs between interpretable and high-performance models

Interpretable models such as Logistic Regression combined with Weight of Evidence transformations provide transparency and are easier to explain to regulators. High-performance models such as Gradient Boosting often achieve higher predictive accuracy but operate as black boxes and can be more difficult to justify in regulated environments.

Organizations must balance predictive performance with explainability, regulatory compliance, fairness, and operational trust.