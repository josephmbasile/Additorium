import PySimpleGUI as sg
import db_calls as db
import datetime
from dateutil import parser
import numpy as np
import matplotlib.pyplot as plt
import csv
from decimal import Decimal as dec
import random
import multiprocessing


#Section 1 Date Information
current_time = datetime.datetime.now()
sample_date = parser.parse('2023-04-02')

current_date = current_time.date()
days_7= current_date - datetime.timedelta(6)
days_30= current_date - datetime.timedelta(30)
days_90= current_date - datetime.timedelta(90)
days_365= current_date - datetime.timedelta(365)




#Section 2 Initial Data

application_title = f"""Additorium Beans"""
avg_error = "TBD"

#Define the classes
weights_classes = [
    "SEKER",
    "BARBUNYA",
    "BOMBAY",
    "CALI",
    "HOROZ",
    "SIRA",
    "DERMASON",
]

training_options = [
    "ALL",
]

for weights_class in weights_classes:
    training_options.append(weights_class)


#Section 3 GUI Layout

sg.theme('LightBrown4')

raw_data_tab = [
    [sg.Text(f"""Use this interface to add training data from the csv file.""", size=(60,2))],
    [sg.Text("Select File:"), sg.Input(enable_events=True, key="-Raw_Data_Input_File-",size=(40,1), default_text=''), sg.FileBrowse(file_types=(("csv files","*.csv"),))],
    [sg.Button(button_text="Generate Training Vectors", key="-Raw_Data_Generate_Training_Vectors-")],
    [sg.Text(f"""Ready to condition input data for training.""", key="-Raw_Data_Messages-")]
]



training_tab = [
    [sg.Text("This tab trains the model based on the training data.")],
    [sg.Text(f"""The most recent error analysis showed an average error of {avg_error}% over 30 samples.""")],
    [sg.Text("Iterations: "), sg.In(default_text="100", key="-Training_Iterations-")],
    [sg.Text("Error Sampling Rate: "), sg.In(default_text="10", key="-Training_Error_Iterations-")],
    [sg.Text("Error Sampling Size: "), sg.In(default_text="30", key="-Training_Error_Samples-")],
    [sg.Text("Learning Rate: "), sg.In(default_text="0.01", key="-Training_Learning_Rate-")],
    [sg.Text("Select Bean Class: "), sg.DropDown(training_options, key="-Class_Selector-", default_value="ALL")],
    [sg.Button("Train", key="-Training_Button-"),sg.Button("View Errors", key="-Training_View_Error_Button-")],
    [sg.Text("Ready to train model.", key="-Training_Messages-")],
    [sg.Text("Initialize Weights: "), sg.In(default_text="0.000000000", key="-Training_Initialize_Value-"), sg.Button("Initialize", key="-Training_Initialize_Button-")],
]

prediction_tab = [
    [sg.Text("This tab will be used to make predictions with the model.")],
    [sg.Button("Evaluate Model", key="-Evaluate_Button-")],
    [sg.Text("Ready to evaluate model.", key="-Evaluate_Messages-")],
]


test_tab = [
    [sg.Text("This tab will be used for troubleshooting the model.")],
    [sg.Text("Test Input 1: "), sg.In(default_text="10", key="-Test_Input-")],
    [sg.Button("Test", key="-Testing_Button-")],
    [sg.Text("Ready to test model.", key="-Training_Messages-")],
]


layout1 = [
    [sg.Text(application_title, auto_size_text=True, size=(60,1), justification="center", font=("",18))],
    [
        sg.TabGroup(
            [
                [
                    sg.Tab('Raw Data',raw_data_tab), #Includes the raw stock data downloaded from Yahoo.
                    #sg.Tab('Training Data', training_data_tab), #Raw stock data is conditioned with moving averages to build input and response vectors.
                    sg.Tab('Training',training_tab), #Input and response vectors are used to train the model.
                    sg.Tab('Prediction',prediction_tab),
                    sg.Tab('Test',test_tab), #The model is used to predict responses using the current model.
                ]
            ]),
    ]
]


#Section 4 Classes and Functions

def generate_training_vectors(values):
    connection = db.create_connection("additorium.db")
    window['-Raw_Data_Messages-'].update("Processing, please wait...")

    #Delete old training data

    delete_training_data = f"""
    DELETE FROM training_data;
    """
    db.execute_query(connection,delete_training_data)

    #Read csv file and insert records into temporary database storage
    csv_file_location = open(values["-Raw_Data_Input_File-"], newline="")
    csv_file = csv.DictReader(csv_file_location, delimiter=",")
    csv_file_tup = []
    for row_1 in csv_file:
        row_tup = (row_1['Area'],row_1['Perimeter'],row_1['MajorAxisLength'],row_1['MinorAxisLength'],row_1['AspectRatio'],row_1['Eccentricity'],row_1['ConvexArea'],row_1['EquivDiameter'],row_1['Extent'],row_1['Solidity'],row_1['Roundness'],row_1['Compactness'],row_1['ShapeFactor1'],row_1['ShapeFactor2'],row_1['ShapeFactor3'],row_1['ShapeFactor4'],row_1['Class'])
        csv_file_tup.append(row_tup)
    #First find the maximum for each column.
    maximums = []
    for col_index in range(len(csv_file_tup[0])):
        maximum = dec(0.00000000)
        if col_index == len(csv_file_tup[0])-1:
            pass
        else:
            for row in csv_file_tup:
                if dec(row[col_index]) > maximum:
                    maximum = dec(row[col_index])
        maximums.append(maximum)
    #print(maximums)
    for row in csv_file_tup:
        #insert the records
        insert_record = f"""
        INSERT INTO training_data (Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4, Class)
        VALUES ("{dec(row[0])/maximums[0]}", "{dec(row[1])/maximums[1]}", "{dec(row[2])/maximums[2]}", "{dec(row[3])/maximums[3]}", "{dec(row[4])/maximums[4]}", "{dec(row[5])/maximums[5]}", "{dec(row[6])/maximums[6]}", "{dec(row[7])/maximums[7]}", "{dec(row[8])/maximums[8]}", "{dec(row[9])/maximums[9]}", "{dec(row[10])/maximums[10]}", "{dec(row[11])/maximums[11]}", "{dec(row[12])/maximums[12]}", "{dec(row[13])/maximums[13]}", "{dec(row[14])/maximums[14]}", "{dec(row[15])/maximums[15]}", "{row[16]}");
        """
        db.execute_query(connection,insert_record)
    window['-Raw_Data_Messages-'].update("Data inserted into database.")


def search_training_data(values):
    """Depreciated"""
    connection = db.create_connection("additorium.db")
    search_term = values["-Training_Data_Search_Term-"]
    retrieve_training_data = f"""
    SELECT id, date, symbol, last_close, last_vol, day_7_avg_close, day_7_avg_vol, day_30_avg_close, day_30_avg_vol, day_90_avg_close, day_90_avg_vol, day_365_avg_close, day_365_avg_vol, target_close
    FROM training_data
    WHERE symbol = "{search_term}";
    """
    training_table_data = db.execute_read_query(connection,retrieve_training_data)
    window['-Training_Data_Table-'].update(training_table_data)

def train_model(input_values):
    values = input_values[0]
    weights_class = input_values[1]
    print("Function started")
    connection = db.create_connection(f"{weights_class}.db")
    training_data_connection = db.create_connection(f"additorium.db")
    iterations = int(values["-Training_Iterations-"])
    error_sampling_rate = int(values["-Training_Error_Iterations-"])
    error_sampling_size = int(values["-Training_Error_Samples-"])
    learning_rate = dec(values["-Training_Learning_Rate-"])

    #Read the training data from the database
    read_training_data = f"""
    SELECT Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4, Class
    FROM training_data
    """
    all_training_data = db.execute_read_query(training_data_connection,read_training_data)

    delete_error_record = f"""
    DELETE FROM training_error
    """
    db.execute_query(connection,delete_error_record)

    get_weights = f"""
    SELECT Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4, bias, id, Class
    FROM weights
    WHERE Class = "{weights_class}"
    ORDER BY id ASC
    """
    #Get a list of tuples from the database.
    all_weights_tup = db.execute_read_query(connection,get_weights)
    #Convert the list of tuples into a list of lists so that they are mutable.
    all_weights = []
    for weight_tup in all_weights_tup:
        these_weights = [
            weight
            for weight in weight_tup
        ]
        all_weights.append(these_weights)


    
    #Iterate
    for iteration in range(iterations):

        #len(random_training_vector) == 11
        weights_delta = dec('0.00000000001')
        

        #print(f"average output: {average_output}")

        random_training_vector = []
        while True:
            
            #print("Alternating vector")
            random_training_index = int(len(all_training_data)*random.random())
            random_training_vector = all_training_data[random_training_index]
            if random_training_vector[-1] == weights_class and iteration % 2 == 0:
                break
            elif random_training_vector[-1] != weights_class and iteration % 2 != 0:
                break
            else:
                pass



        #print(random_training_vector)
        
        derror_dweights = []

        
        #Determine the target by checking for a matching class label
        target = dec(0.0000)
        if random_training_vector[-1] == weights_class:
            target = dec(1.0000)
        #print(target)
        current_error = (compute_network(all_weights,random_training_vector)-target)**2
        if current_error > 0.0001:
            #Iterate through the weights in each class.
            for i in range(len(all_weights_tup)):
                derror_dweights.append([])
                for j in range(len(all_weights_tup[i])-2):
                    #Calculate the low weight and compute the error
                    
                    
                    #print(weights_low)
                    weights_low = get_weights_low(all_weights, i, j, weights_delta)
                    #print(f"weights_low[i][j]: {weights_low[i][j]}")
                    #all_weights[i][j] = dec(weights_low) - weights_delta
                    output_low = compute_network(weights_low,random_training_vector)

                    error_low = (output_low - dec(target))**2
                    #print(f"Output low: {output_low}")
                    #print(f"error low: {error_low}")
                    
                    #Calculate the high weight and compute the error
                    #print(all_weights[i][j])
                    weights_high = get_weights_high(all_weights, i, j, weights_delta)
                    #print(f"weights_high[i][j]: {weights_high[i][j]}")
                    #all_weights[i][j] = dec(all_weights[i][j]) + 2*weights_delta
                    output_high = compute_network(weights_high,random_training_vector)
                    error_high = (output_high - dec(target))**2
                    #print(f"Output high: {output_high}")
                    #print(f"error high: {error_high}")

                    #Calculate the gradient for the weight and append it to the list
                    derror_dweight = (error_high - error_low)/(2*weights_delta)
                    #print(f"derror_dweight: {derror_dweight}")
                    #print(f"i: {i}")
                    derror_dweights[i].append(derror_dweight)

                #Calculate the low bias and compute the error
                bias_low = get_weights_low(all_weights, i, -3, weights_delta)
                output_low = compute_network(bias_low,random_training_vector)
                error_bias_low = (output_low - dec(target))**2

                #Calculate the high bias and compute the error
                bias_high = get_weights_high(all_weights, i, -3, weights_delta)
                output_high = compute_network(bias_high,random_training_vector)
                error_bias_high = (output_high - dec(target))**2

                #Calculate the gradient for the bias and append it to the list
                derror_dbias = (error_bias_high - error_bias_low)/(2*weights_delta)
                #print(derror_dbias)
                derror_dweights[i].append(derror_dbias)
                #Append the row id
                derror_dweights[i].append(all_weights_tup[i][-2])
            
                #print(f"derror_dweights: {derror_dweights}")

                #update the weights for the next iteration (not in the db)
                all_weights = update_weights_iterator(all_weights, derror_dweights, learning_rate, i, current_error)

            #Calculate the error sample and record to the database
            if iteration % error_sampling_rate == 0 and iteration > 2:
                error_sample, average_output = get_error_sample(weights_class, error_sampling_size, all_training_data)
                percent_done = round((iteration / iterations)*100,2)
                print(f"""{error_sample}; {percent_done}%""")
                insert_error_record = f"""
                INSERT INTO training_error (cumulative_error, samples, iteration)
                VALUES ("{error_sample}", "{error_sampling_size}", "{iteration}")
                """
                db.execute_query(connection,insert_error_record)

                #Record the new weights in the database
                for r in range(len(all_weights)):
                    #print('all weights, final')
                    #print(all_weights)
                    update_weights = f"""
                    UPDATE weights
                    SET Area="{all_weights[r][0]}", Perimeter="{all_weights[r][1]}", MajorAxisLength="{all_weights[r][2]}", MinorAxisLength="{all_weights[r][3]}", AspectRatio="{all_weights[r][4]}", Eccentricity="{all_weights[r][5]}", ConvexArea="{all_weights[r][6]}", EquivDiameter="{all_weights[r][7]}", Extent="{all_weights[r][8]}", Solidity="{all_weights[r][9]}", Roundness="{all_weights[r][10]}", Compactness="{all_weights[r][11]}", ShapeFactor1="{all_weights[r][12]}", ShapeFactor2="{all_weights[r][13]}", ShapeFactor3="{all_weights[r][14]}", ShapeFactor4="{all_weights[r][15]}", bias="{all_weights[r][16]}", Class="{all_weights[r][18]}"
                    WHERE id = {all_weights[r][-2]}; 
                    """
                    db.execute_query(connection,update_weights)
                    #print(updated_weights)



    
def get_weights_low(all_weights_tup, i, j,weights_delta):
    all_weights = []
    for weight_tup in all_weights_tup:
        these_weights = [
            weight
            for weight in weight_tup
        ]
        all_weights.append(these_weights)
    all_weights[i][j] = f"""{dec(all_weights[i][j]) - dec(weights_delta)}"""
    return all_weights

def get_weights_high(all_weights_tup, i, j,weights_delta):
    all_weights = []
    for weight_tup in all_weights_tup:
        these_weights = [
            weight
            for weight in weight_tup
        ]
        all_weights.append(these_weights)
    all_weights[i][j] = f"""{dec(all_weights[i][j]) + dec(weights_delta)}"""
    return all_weights

def get_error_sample(weights_class, error_sampling_size, all_training_data):
    connection = db.create_connection(f"{weights_class}.db")
    cumulative_error = 0
    average_output = []
    cumulative_output = 0
    for i in range(error_sampling_size):

        get_weights = f"""
        SELECT Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4, bias, id, Class
        FROM weights
        WHERE Class = "{weights_class}"
        ORDER BY id ASC
        """
        #Get a list of tuples from the database.
        all_weights_tup = db.execute_read_query(connection,get_weights)
        #print(f"ALL WEIGHTS: {all_weights_tup}")
        #Convert the list of tuples into a list of lists so that they are mutable.
        all_weights = []
        for weight_tup in all_weights_tup:
            these_weights = [
                weight
                for weight in weight_tup
            ]
            all_weights.append(these_weights)
        random_training_vector = []
        while True:
            random_training_index = int(len(all_training_data)*random.random())
            random_training_vector = all_training_data[random_training_index]
            if random_training_vector[-1] == weights_class and i % 2 == 0:
                break
            elif random_training_vector[-1] != weights_class and i % 2 != 0:
                break
            else:
                pass
        target = dec(0.0000)
        if random_training_vector[-1] == weights_class:
            target = dec(1.00000)
        compute_output = compute_network(all_weights, random_training_vector)
        this_error = compute_output - dec(target)
        print(f"This class: {weights_class}; target: {target}; Output: {compute_output}")
        cumulative_error = (cumulative_error + abs(this_error))
        cumulative_output = cumulative_output + compute_output
    this_average_output = cumulative_output / error_sampling_size
    average_output.append(this_average_output)
    cumulative_error = cumulative_error / (error_sampling_size)
    #print(average_output)
    return cumulative_error, average_output

def update_weights_iterator(all_weights_tup, derror_dweights, learning_rate, i, current_error):
    #print("All weights, before update")
    #print(all_weights)
    #print(derror_dweights)
    #print(all_weights[i][3])
    effective_learning_rate = learning_rate#*current_error
    all_weights = []
    for weight_tup in all_weights_tup:
        these_weights = [
            weight
            for weight in weight_tup
        ]
        all_weights.append(these_weights)
    for j in range(len(all_weights[i])):
        if j == 17 or j == 18:
            pass
        else:
            #print(f"i: {i}")
            #print(f"length all_weights{len(all_weights[i])}")
            #print(f"length weights{len(weights)}")
            #print(f"length derror_dweights{len(derror_dweights[i][j])}")
            if j == 16:
                all_weights[i][j] = dec(all_weights[i][j]) - dec(derror_dweights[i][j])*dec(effective_learning_rate)*1 
            else:
                all_weights[i][j] = dec(all_weights[i][j]) - dec(derror_dweights[i][j])*dec(effective_learning_rate)

    #print(all_weights)
    return all_weights
    

def compute_network(weights,vector):
    layer1 = compute_layer_1(weights,vector)
    layer2 = compute_layer_2(weights,vector,layer1)
    layer3 = compute_layer_3(weights,vector,layer2)
    layer4 = compute_layer_4(weights,vector,layer3)
    layer5 = compute_layer_5(weights,vector,layer4)
    layer6 = compute_layer_6(weights,vector,layer5)
    layer7 = compute_layer_7(weights,vector,layer6)
    layer8 = compute_layer_8(weights,vector,layer7)
    layer9 = compute_layer_9(weights,vector,layer8)
    layer10 = compute_layer_10(weights,vector,layer9)
    layer11 = compute_layer_11(weights,vector,layer10)
    layer12 = compute_layer_12(weights,layer6)

    return layer12

def compute_layer_1(weights,vector):
    output = []
    for i in range(len(vector)-1):
        this_output = f"{(dec(weights[0][i]))*(dec(vector[i])**10)}"
        output.append(this_output)
    return output

def compute_layer_2(weights,vector, layer1):
    output = []
    for i in range(len(vector)-1):
        this_output = f"{(dec(weights[1][i]))*(dec(vector[i])**9)+dec(layer1[i])}"
        output.append(this_output)
    return output

def compute_layer_3(weights,vector, layer2):
    output = []
    for i in range(len(vector)-1):
        this_output = f"{(dec(weights[2][i]))*(dec(vector[i])**8)+dec(layer2[i])}"
        output.append(this_output)
    return output

def compute_layer_4(weights,vector, layer3):
    output = []
    for i in range(len(vector)-1):
        this_output = f"{(dec(weights[3][i]))*(dec(vector[i])**7)+dec(layer3[i])}"
        output.append(this_output)
    return output

def compute_layer_5(weights,vector, layer4):
    output = []
    for i in range(len(vector)-1):
        this_output = f"{(dec(weights[4][i]))*(dec(vector[i])**6)+dec(layer4[i])}"
        output.append(this_output)
    return output

def compute_layer_6(weights,vector, layer5):
    output = []
    for i in range(len(vector)-1):
        this_output = f"{(dec(weights[5][i]))*(dec(vector[i])**5)+dec(layer5[i])}"
        output.append(this_output)
    return output

def compute_layer_7(weights, vector, layer6):
    output = []
    for i in range(len(vector)-1):
        output.append(f"{dec(weights[6][i])*(dec(vector[i])**4)+dec(layer6[i])}")
    return output

def compute_layer_8(weights,vector, layer7):
    output = []
    for i in range(len(vector)-1):
        output.append(f"{dec(weights[7][i])*(dec(vector[i])**3)+dec(layer7[i])}")
    return output

def compute_layer_9(weights,vector, layer8):

    output = []
    for i in range(len(vector)-1):
        output.append(f"{dec(weights[8][i])*(dec(vector[i])*dec(vector[i]))+dec(layer8[i])}")
    return output

def compute_layer_10(weights, vector, layer9):
    #print(f"layer4: {layer4}")
    output = []
    for i in range(len(vector)-1):
        output.append(f"{dec(weights[9][i])*(dec(vector[i]))+dec(layer9[i])}")
    #print(f"layer 5 output: {output}")
    return output
    
def compute_layer_11(weights, vector, layer10):
    output = []
    for i in range(len(vector)-1):
        output.append(f"{dec(weights[10][i])+dec(layer10[i])}")
    #print(f"layer 6 output: {output}")
    return output

def compute_layer_12(weights,layer11):
    output = dec(0.0000000000000000000000000)
    #print(len(weights))
    for i in range(len(weights[6])-3):
        #print(weights[i])
        output = output + (dec(weights[11][i])*(dec(layer11[i])))
        #print(f"{weights[3][i]}; {layer6[i]}")
    #print(f"layer 7 dot product: {output}")
    output = output + dec(weights[6][-3]) #add bias
    #print(f"Layer7 intermediate: {output}")
    #Apply logistic function
    output = 1 / (1+ np.exp(-output/dec(1)))
    #print(f"target: {layer5[10]}")
    #print(f"layer 7 output: {output}")
    return output


    


def generate_input_vector(values,prediction_date):
    pass

def generate_actual_response(values,date):
    pass

def initialize_weights(values, weights_classes):
    for weights_class in weights_classes:
        connection = db.create_connection(f"{weights_class}.db")
        initial_value = values["-Training_Initialize_Value-"]
        read_weights = f"""
        SELECT *
        FROM weights
        """
        all_weights_tup = db.execute_read_query(connection,read_weights)
        for weights in all_weights_tup:
            write_weights = f"""
            UPDATE weights
            SET Area="{initial_value}", Perimeter="{initial_value}", MajorAxisLength="{initial_value}", MinorAxisLength="{initial_value}", AspectRatio="{initial_value}", Eccentricity="{initial_value}", ConvexArea="{initial_value}", EquivDiameter="{initial_value}", Extent="{initial_value}", Solidity="{initial_value}", Roundness="{initial_value}", Compactness="{initial_value}", ShapeFactor1="{initial_value}", ShapeFactor2="{initial_value}", ShapeFactor3="{initial_value}", ShapeFactor4="{initial_value}", bias="{initial_value}", Class="{weights[-1]}"
            WHERE id="{weights[0]}"
            """
            db.execute_query(connection,write_weights)
        window['-Training_Messages-'].update(f"Weights Initialized at {initial_value}")

def initialize_weights_class(values, weights_class):
    
    connection = db.create_connection(f"{weights_class}.db")
    initial_value = values["-Training_Initialize_Value-"]
    read_weights = f"""
    SELECT *
    FROM weights
    """
    all_weights_tup = db.execute_read_query(connection,read_weights)
    for weights in all_weights_tup:
        write_weights = f"""
        UPDATE weights
        SET Area="{initial_value}", Perimeter="{initial_value}", MajorAxisLength="{initial_value}", MinorAxisLength="{initial_value}", AspectRatio="{initial_value}", Eccentricity="{initial_value}", ConvexArea="{initial_value}", EquivDiameter="{initial_value}", Extent="{initial_value}", Solidity="{initial_value}", Roundness="{initial_value}", Compactness="{initial_value}", ShapeFactor1="{initial_value}", ShapeFactor2="{initial_value}", ShapeFactor3="{initial_value}", ShapeFactor4="{initial_value}", bias="{initial_value}", Class="{weights[-1]}"
        WHERE id="{weights[0]}"
        """
        db.execute_query(connection,write_weights)
    window['-Training_Messages-'].update(f"{weights_class} Weights Initialized at {initial_value}")



def run_test_function(values):
    """Performs test cases using the other functions."""
    #Retrieve a set of weights from the database
    connection = db.create_connection("additorium.db")

    read_training_data = f"""
    SELECT Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4, Class
    FROM training_data
    """
    all_training_data = db.execute_read_query(connection,read_training_data)

    
    
    #random_training_index = int(len(all_training_data)*random.random())
    random_training_index = 155
    random_training_vector = all_training_data[random_training_index]

    #print(random_training_vector)
    get_weights = f"""
    SELECT Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4, bias, id, Class
    FROM weights
    WHERE Class = "SEKER"
    ORDER BY id ASC
    """
    #print(weights_classes[k])
    #Get a list of tuples from the database.
    all_weights_tup = db.execute_read_query(connection,get_weights)

    #print(all_weights_tup)

    output = compute_network(all_weights_tup,random_training_vector)
    #output = compute_layer_4(all_weights_tup,random_training_vector)
    print(output)

def evaluate_model(values,weights_classes):
    all_weights = []
    for bean_class in weights_classes:
        this_connection = db.create_connection(f"{bean_class}.db")
        read_weights = f"""
        SELECT Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4, bias, id, Class
        FROM weights
        WHERE Class="{bean_class}"
        ORDER BY id ASC;
        """
        these_weights = db.execute_read_query(this_connection,read_weights)
        all_weights.append(these_weights)
        db.close_connection(this_connection)
    percent_correct = []
    for index in range(len(weights_classes)):
        connection = db.create_connection("additorium.db")

        read_input_vectors = f"""
        SELECT Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4, Class 
        FROM training_data
        WHERE class = "{weights_classes[index]}";
        """
        these_vectors = db.execute_read_query(connection,read_input_vectors)
        total_correct = 0
        
        for vector in these_vectors:
            outputs =[]
            for weights in all_weights:
                network_output = compute_network(weights,vector)
                outputs.append(network_output)
            output_maximum = 0
            maximum_index = 0
            for index2 in range(len(outputs)):
                if outputs[index2] > output_maximum:
                    output_maximum = outputs[index2]
                    maximum_index = index2
            #print(maximum_index)
            if weights_classes[maximum_index] == vector[-1]:
                total_correct = total_correct +1
        percent = round((total_correct / len(these_vectors))*100,2)
        percent_correct.append(percent)
    window["-Evaluate_Messages-"].update(f"""
    {weights_classes[0]} had a {percent_correct[0]}% success rate.
    {weights_classes[1]} had a {percent_correct[1]}% success rate.
    {weights_classes[2]} had a {percent_correct[2]}% success rate.
    {weights_classes[3]} had a {percent_correct[3]}% success rate.
    {weights_classes[4]} had a {percent_correct[4]}% success rate.
    {weights_classes[5]} had a {percent_correct[5]}% success rate.
    {weights_classes[6]} had a {percent_correct[6]}% success rate.
    """)


def train_model_multi(values):
    input_values = []
    for weights_class in weights_classes:
        input_values.append([values,weights_class])
    with multiprocessing.Pool(processes=7) as pool:
        pool.map(train_model,input_values)

#Section 5 Window and Event Loop
window = sg.Window(title=application_title, layout= layout1, margins=(25,25))

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    elif event == "-Raw_Data_Generate_Training_Vectors-":
        generate_training_vectors(values)
    elif event == "-Training_Data_Search_Button-":
        search_training_data(values)
    elif event == "-Training_Button-":
        weights_class = values["-Class_Selector-"]
        if weights_class == "ALL" and __name__ == "__main__":
            train_model_multi(values)
        else:
            train_model([values, weights_class])
    elif event == "-Training_Initialize_Button-":
        weights_class = values["-Class_Selector-"]
        if weights_class == "ALL":
            initialize_weights(values, weights_classes)
        else:
            initialize_weights_class(values, weights_class)
    elif event == "-Testing_Button-":
        run_test_function(values)
    elif event == "-Evaluate_Button-":
        evaluate_model(values, weights_classes)
window.close()