import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_ta as ta
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

def prepare_data(ticker, hist_prices):
    print(f"\nprepare_data called")

    df = hist_prices.copy()

    # Generate Technical Indicators
    # 'ta.AllStudy' generates a set of 337 indicators as of 3 Feb 2026
    df.ta.study(ta.AllStudy, cores=0)
    print(f"Total features generated: {len(df.columns)}")

    # Define Target Variable Gamma(t)
    # Gamma(t) = 1 if Open(t+1) - Open(t) > 0, else -1
    # We shift 'open' by -1 to compare today's open with tomorrow's open
    df['target'] = np.where(df['open'].shift(-1) > df['open'], 1, -1)

    print("\ntarget: ", df['target'])

    # Remove indicators with > 10% NaNs (to avoid losing too much history)
    nan_threshold = 0.10 * len(df)
    cols_to_drop = df.columns[df.isna().sum() > nan_threshold].tolist()
    print("\n Columns to drop: ", cols_to_drop)
    print(f"Numbers of dropped columns: {len(cols_to_drop)}")
    df.drop(columns=cols_to_drop, inplace=True)

    # Drop rows with any remaining NaNs
    initial_len = len(df)
    df.dropna(inplace=True)
    print(f"\nRows dropped during cleaning: {initial_len - len(df)}")
    print(f"\nPercentage rows dropped: {(initial_len - len(df))*100/initial_len}")

    # Split into Features (X) and Target (y)
    X = df.drop(columns=['target'])
    y = df['target']

    print(f"\nprepare_data out")
    return X, y

def cross_validation(ticker, X, y):
    print("\ncross_validation called")

    # Define a pipeline that processes sequentially a min-max normalisation and MLP classifier
    hidden_layer_sizes = int((X.shape[1] + len(np.unique(y)))/2)
    print("\n hidden_layer_sizes: ", hidden_layer_sizes)

    pipeline = Pipeline([
        ('scaler', MinMaxScaler(feature_range=(0, 1))),
        ('mlp', MLPClassifier(  hidden_layer_sizes=hidden_layer_sizes,
                                activation='logistic',
                                solver='lbfgs',
                                batch_size='auto',
                                learning_rate='adaptive',
                                learning_rate_init=0.03,
                                max_iter=5000,
                                momentum=0.2,
                                random_state=np.random.get_state()[1][0],
                                early_stopping=False)
        )
    ])

    # Define a 10-fold cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=False)
    # List to store the rows for Table 4
    results_data = []
    # Initialize current dataset (starts with all features)
    X_current = X.copy()

    # We first use the 337 and average the accuracy
    print("\nEvaluating for n=0")
    base_acc = cross_val_score(pipeline, X, y, cv=skf).mean()
    print("Accuracy: ", base_acc*100)

    # Regroup the results
    results_data.append({
        "n": "0",
        "Features": X.shape[1],
        "Accuracy": base_acc
    })
    # Set an arbitrary maximum loop to avoid running the code for too long
    max_n=10
    # Start looping for n=1 to max_n
    for i in range(1, max_n + 1):
        print(f"\n--- Processing n={i} ---")

        # We choose 25% features using the Pearson correlation coefficient
        correlations = X_current.corrwith(y).abs()

        # Extract the 5 most important features
        top_5 = correlations.sort_values(ascending=False).head(5)
        print(f"\nTop 5 Salient Features: {top_5}")

        # Calculate cutoff (top 25%)
        cutoff = correlations.quantile(0.75)
        print(f"Cutoff: {cutoff}")

        # Give the most important features
        salient_features = correlations[correlations >= cutoff].index.tolist()

        # Avoid empty features set
        if len(salient_features) <= 1:
            print("No features left. Stopping iteration.")
            break

        # Update the dataset for the current iteration
        X_current = X[salient_features]
        print("\nX_selected shape: ", X_current.shape)

        print(f"Evaluating n={i}: ({len(salient_features)} features)...")
        salient_acc = cross_val_score(pipeline, X_current, y, cv=skf).mean()

        # Regroup the results
        results_data.append({
            "n": str(i),
            "Features": len(salient_features),
            "Accuracy": salient_acc
        })

        # Print a summary table, update at each step
        summary_table = pd.DataFrame(results_data)
        print("\nSummary")
        print(summary_table.to_string(index=False))

    # Create final DataFrame for plotting
    final_df = pd.DataFrame(results_data)

    # Calculate Accuracy Gain relative to n=0 (all features)
    # Formula: (Current_Acc - Base_Acc) * 100
    base_val = final_df.iloc[0]['Accuracy']
    final_df['Accuracy Gain'] = (final_df['Accuracy'] - base_val) * 100

    plt.figure(figsize=(8, 5))

    # Plot line chart with markers
    plt.plot(final_df['n'], final_df['Accuracy Gain'],
             marker='o', linestyle='-', color='navy', linewidth=2, markersize=8)

    # Formatting
    plt.ylabel("Accuracy gain (%)")
    plt.xlabel("Iteration (n)")
    plt.title(f"Accuracy gain for {ticker}")

    # Add zero line for reference
    plt.axhline(0, color='black', linewidth=1)

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save and Show
    os.makedirs("outputs", exist_ok=True) # Ensure folder exists, create it otherwise
    plt.savefig(f"outputs/{ticker}_accuracy_gain.pdf")
    plt.show()

    print("\ncross_validation out")
    return salient_features


if __name__ == '__main__':
    os.makedirs("data", exist_ok=True) # Ensure folder exists, create it otherwise
    price_history_path = 'data/price_history.csv'
    hist_prices = pd.DataFrame()

    if not os.path.exists(price_history_path):
        hist_prices = yf.download("ECH", start="2009-12-12", end="2020-01-01", auto_adjust=True, progress=False)
        hist_prices.to_csv(price_history_path, index=True)

    hist_prices = pd.read_csv(price_history_path, index_col=0, parse_dates=True, header=[0, 1]) # Remove "Ticker" from headers
    hist_prices.columns = [col[0].lower() for col in hist_prices.columns] # Use lowercase

    print("hist_prices: ", hist_prices.head(5))
    print(f"\nColumns: {hist_prices.columns.tolist()}")

    X, y = prepare_data("ECH", hist_prices) # Prepare the data
    print("\n X: ", X)
    print("\n y: ", y)

    cv = cross_validation("ECH", X, y)
    print("\ncv: ", cv)
