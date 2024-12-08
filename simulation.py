import csv
import path_planning as pp

def test_search_algorithms(iterations, output_file):
    results = []
    for i in range(iterations):
        # Generate new mazes for each iteration
        #_map_ = pp.generateMap2d([60, 60])
        map_h_object, info = pp.generateMap2d_obstacle([60, 60])
        #pp._map_ = _map_
        pp.map_h_object = map_h_object

        # Run the main function from path_planning.py and collect the results
        iteration_results = pp.main()
        for result in iteration_results:
            result["iteration"] = i + 1
            results.append(result)
    return results

def save_results_to_csv(results, output_file):
    # Save the results to a CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

# Run the test_search_algorithms function 100 times and save the results to a CSV file
results = test_search_algorithms(100, "results_w_H_Obstacle.csv")
save_results_to_csv(results, "results_w_H_Obstacle.csv")