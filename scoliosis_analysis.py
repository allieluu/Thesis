from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import seaborn as sns


def calculate_deltas(snps: pd.DataFrame) -> pd.DataFrame:
    columns = ["#rsID", "genotype", "delta", "chr"]
    global delta_rsIDs
    index = 0
    print("Analyzing chromosome {}".format(
        snps.get_value(snps.index.values[index], "chr")))
    for idx, row in snps.iterrows():
        # print(row)
        # print(row[0])
        deltas = [row[0] - row[3], row[1] - row[4], row[2] - row[5]]
        # print(deltas)
        max_delta = np.argmax(deltas)
        if max_delta == 0:
            genotype = "11"
        elif max_delta == 1:
            genotype = "12"
        else:
            genotype = "22"
        temp_array = np.array([idx, genotype, deltas[max_delta], snps.get_value(
            snps.index.values[index], "chr")])
        print(temp_array)
        temp_df = pd.DataFrame(data=temp_array)
        delta_rsIDs.append(temp_df)
        index += 1
    print("Finished analyzing {}".format(
        snps.get_value(snps.index.values[0], "chr")))
    print(delta_rsIDs.columns)
    return delta_rsIDs


def main():
    # read in csv file make data frame with rsIDs as rows;
    # case/control11, case/control12, case/control22,
    # case/control??
    #
    # make sample weight matrix
    #
    # make target values matrix
    np.set_printoptions(suppress=True)
    columns_to_use = ["#rsID", "case11", "case12", "case22", "ctrl11", "ctrl12",
                      "ctrl22", "chr"]
    df = pd.read_csv(
        "../../tigress/arburton/scoliosis_data/scoliosis_data/"
        "ReleaseN12_AffectVsUnivN2_All-20141224.csv",
        # "/Volumes/Transcend/IWFall16/scoliosis_data"
        # "/ReleaseN12_AffectVsUnivN2_All-20141224.csv",
        index_col=0,
        usecols=columns_to_use,
        dtype={'#rsID': str})
    # get length of df then create target values matrix of 3 columns of 1's
    # and 3 columns of 2's
    num_rows = len(df)

    # TODO: use sample weights as target values with RandomForestRegressor
    target_values_ctrl = np.zeros((num_rows, 3))
    target_values_case = np.ones((num_rows, 3))
    target_values = np.hstack((target_values_case, target_values_ctrl))

    # X = df.values
    # sample_weights = np.copy(X).astype(dtype="float32")
    # print(sample_weights)
    # sample_weights[:, 0:3] = sample_weights[:, 0:3] / 853
    # sample_weights[:, 3:6] = sample_weights[:, 3:6] / 7799
    # print(sample_weights)
    df.iloc[:, 0:3] = df.iloc[:, 0:3].apply(pd.to_numeric)
    df.iloc[:, 3:6] = df.iloc[:, 3:6].apply(pd.to_numeric)

    df.iloc[:, 0:3] = df.iloc[:, 0:3].divide(853)
    df.iloc[:, 3:6] = df.iloc[:, 3:6].divide(7799)
    print(df.head(n=5))
    grouping = df.groupby('chr')

    # np.true_divide(sample_weights[:, 1:4], 1250, out=sample_weights,
    #                casting="unsafe")

    # delta_vals = pd.DataFrame(index=X['#rsID'], columns={'#rsID', '11', '12',
    #                                                      '22'})
    # for index, row in df.iterrows():
    # mean = np.mean(sample_weights)
    # print(mean)
    # TODO: construct delta dataframe
    print("Calculating deltas...")
    global delta_rsIDs
    delta_rsIDs = pd.DataFrame(index=df.index.values, columns={"#rsID",
                                                               "genotype",
                                                               "delta",
                                                               "chr"})
    # TODO: return all values that are above the mean
    # Split dataframe

    # index = 0
    # for row in sample_weights:
    #     print("Analyzing rsID {}".format(df.index.values[index]))
    #     deltas = [row[0] - row[3], row[1] - row[4], row[2] - row[5]]
    #     max_delta = np.argmax(deltas)
    #     if max_delta == 0:
    #         genotype = "11"
    #     elif max_delta == 1:
    #         genotype = "12"
    #     else:
    #         genotype = "22"
    #     delta_rsIDs.add({df.index.values[index], genotype, deltas[max_delta],
    #                      df.get_value(df.index.values[index], "chr")})
    #     index += 1
    # TODO: this is entirely wrong
    Parallel(n_jobs=2, verbose=60)(delayed(calculate_deltas)
                                   (group)
                                   for name, group in grouping)

    print("Columns of delta_rsIDs post processing: ")
    print(delta_rsIDs.columns)

    delta_rsIDs.set_index("#rsID")
    delta_rsIDs.sort(columns="delta", ascending=False)
    print(delta_rsIDs.head(n=25))

    delta_rsIDs.to_pickle("../../tigress/arburton/scoliosis_data"
                          "/scoliosis_data/rs_deltas.pkl")
    # plot counts from chromosomes of top 25 rsIDs
    ax = sns.countplot(x="chr", data=delta_rsIDs.head(n=25))
    ax.set(xlabel="Chromosome", ylabel="Number of SNPs")
    ax.figure.savefig("../../tigress/arburton/countplot_scoliosis.png")

    # TODO: generate graphs
    # TODO: countplot/barplot by chromosome
    # TODO: manhattan plot --> use seaborn stripplot w/ jitter=True
    ax = sns.stripplot(x="chr", y="delta", jitter=True)
    ax.set(xlabel="Chromosome", ylabel="Percent above Control")
    ax.figure.savefig("../../tigress/arburton/manhattan_plot_scoliosis.png")


# TODO: Run regression models with just case variables
# --> target values = num_patients/total cases

if __name__ == '__main__':
    delta_rsIDs = pd.DataFrame()
    main()
