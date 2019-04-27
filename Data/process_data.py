import sys
import pandas as pd
import sqlalchemy


def load_data(messages_filepath, categories_filepath):

    #Reading files

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    #Mergind dataframes

    df = messages.merge(categories, how='left', on='id')

    return df


def clean_data(df):

    # Splitting categories into separate columns
    categories = df['categories'].str.split(';', expand=True)

    # Extracting a list of column names for categories
    # Removing irrelevant characters from each category
    row = categories[:1]
    category_colnames = row.apply(lambda x: x.str[:-2]).values

    # Renaming the columns of `categories`
    categories.columns = category_colnames[0]

    # Converting category values to just number 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = pd.to_numeric(categories[column])

    # Dropping original categories columns from merged dataset
    df.drop(columns='categories', inplace=True)

    # Concatenating the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Removing duplicates
    df.drop_duplicates(inplace=True)

    # DEFINING FEATURES

    #X = df['message'].values
    #y = df.drop(columns=['id', 'message', 'original', 'genre']).values

    return df



def save_data(df,database_name):
    # LOADING TO DATABASE

    database_filepath = 'sqlite:///{}.db'.format(database_name)
    engine = sqlalchemy.create_engine(database_filepath)
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()