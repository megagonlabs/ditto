import pandas as pd

def topK_neighbors_to_candidate_set(topK_neighbors):
    #We create a data frame corresponding to topK neighbors.
    # We are given a 2D matrix of the form 1: [a1, a2, a3], 2: [b1, b2, b3]
    # where a1, a2, a3 are the top-3 neighbors for tuple 1 and so on.
    # We will now create a two column DF fo the form (1, a1), (1, a2), (1, a3), (2, b1), (2, b2), (2, b3)
    topK_df = pd.DataFrame(topK_neighbors)
    topK_df["ltable_id"] = topK_df.index
    melted_df = pd.melt(topK_df, id_vars=["ltable_id"])
    melted_df["rtable_id"] = melted_df["value"]
    candidate_set_df = melted_df[["ltable_id", "rtable_id"]]
    return candidate_set_df


#This accepts four inputs:
# data frames for candidate set and ground truth matches
# left and right data frames
def compute_blocking_statistics(candidate_set_df, golden_df, left_df, right_df):
    #Now we have two data frames with two columns ltable_id and rtable_id
    # If we do an equi-join of these two data frames, we will get the matches that were in the top-K
    print(candidate_set_df.head(),golden_df.head())
    #merged_df = pd.merge(candidate_set_df, golden_df, on=['ltable_id', 'rtable_id'])
    merged_df = pd.merge(candidate_set_df, golden_df, on=['ltable_id', 'rtable_id'])

    left_num_tuples = len(left_df)
    right_num_tuples = len(right_df)
    statistics_dict = {
        "left_num_tuples": left_num_tuples,
        "right_num_tuples": right_num_tuples,
        "recall": len(merged_df) / len(golden_df),
        "cssr": len(candidate_set_df) / (left_num_tuples * right_num_tuples)
        }

    return statistics_dict


#This function is useful when you download the preprocessed data from DeepMatcher dataset
# and want to convert to matches format.
#It loads the train/valid/test files, filters the duplicates,
# and saves them to a new file called matches.csv
def process_files(folder_root):
    df1 = pd.read_csv(folder_root + "/train.csv")
    df2 = pd.read_csv(folder_root + "/valid.csv")
    df3 = pd.read_csv(folder_root + "/test.csv")

    df1 = df1[df1["label"] == 1]
    df2 = df2[df2["label"] == 1]
    df3 = df3[df3["label"] == 1]

    df = pd.concat([df1, df2, df3], ignore_index=True)

    df[["ltable_id","rtable_id"]].to_csv(folder_root + "/matches.csv", header=True, index=False)
