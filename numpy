{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOUYudN3KsGYmCe2skYOGsr",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/sangeethsanthosh-git/AEROSENSE/blob/main/numpy\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "id": "O8bKaS5LV14P",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d31dc873-f891-403e-98ef-9eac8ce63402"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mean MPG: 23.514572864321607\n",
            "Median MPG: 23.0\n",
            "Std MPG: 7.806159061274433\n",
            "Cars with MPG > 25: 158\n",
            "\n",
            "Cars with >6 cylinders:\n",
            "['chevrolet chevelle malibu', 'buick skylark 320', 'plymouth satellite', 'amc rebel sst', 'ford torino', 'ford galaxie 500', 'chevrolet impala', 'plymouth fury iii', 'pontiac catalina', 'amc ambassador dpl', 'dodge challenger se', \"plymouth 'cuda 340\", 'chevrolet monte carlo', 'buick estate wagon (sw)', 'ford f250', 'chevy c20', 'dodge d200', 'hi 1200d', 'chevrolet impala', 'pontiac catalina brougham', 'ford galaxie 500', 'plymouth fury iii', 'dodge monaco (sw)', 'ford country squire (sw)', 'pontiac safari (sw)', 'chevrolet impala', 'pontiac catalina', 'plymouth fury iii', 'ford galaxie 500', 'amc ambassador sst', 'mercury marquis', 'buick lesabre custom', 'oldsmobile delta 88 royale', 'chrysler newport royal', 'amc matador (sw)', 'chevrolet chevelle concours (sw)', 'ford gran torino (sw)', 'plymouth satellite custom (sw)', 'buick century 350', 'amc matador', 'chevrolet malibu', 'ford gran torino', 'dodge coronet custom', 'mercury marquis brougham', 'chevrolet caprice classic', 'ford ltd', 'plymouth fury gran sedan', 'chrysler new yorker brougham', 'buick electra 225 custom', 'amc ambassador brougham', 'chevrolet impala', 'ford country', 'plymouth custom suburb', 'oldsmobile vista cruiser', 'chevrolet monte carlo s', 'pontiac grand prix', 'dodge dart custom', 'oldsmobile omega', 'ford gran torino', 'buick century luxus (sw)', 'dodge coronet custom (sw)', 'ford gran torino (sw)', 'amc matador (sw)', 'pontiac catalina', 'chevrolet bel air', 'plymouth grand fury', 'ford ltd', 'chevrolet monza 2+2', 'ford mustang ii', 'chevrolet chevelle malibu classic', 'dodge coronet brougham', 'amc matador', 'ford gran torino', 'plymouth volare premier v8', 'cadillac seville', 'chevy c10', 'ford f108', 'dodge d100', 'chevrolet caprice classic', 'oldsmobile cutlass supreme', 'dodge monaco brougham', 'mercury cougar brougham', 'pontiac grand prix lj', 'chevrolet monte carlo landau', 'chrysler cordoba', 'ford thunderbird', 'oldsmobile cutlass salon brougham', 'dodge diplomat', 'mercury monarch ghia', 'chevrolet monte carlo landau', 'ford futura', 'dodge magnum xe', 'chevrolet caprice classic', 'ford ltd landau', 'mercury grand marquis', 'dodge st. regis', 'buick estate wagon (sw)', 'ford country squire (sw)', 'chevrolet malibu classic (sw)', 'chrysler lebaron town @ country (sw)', 'cadillac eldorado', 'oldsmobile cutlass salon brougham', 'oldsmobile cutlass ls']\n",
            "\n",
            "Horsepower increased by 10%:\n",
            "[143.  181.5 165.  165.  154.  217.8 242.  236.5 247.5 209. ]\n",
            "\n",
            "Average displacement (Europe): 109.14285714285714\n",
            "\n",
            "Mean MPG by cylinders:\n",
            "3 cylinders → 20.55\n",
            "4 cylinders → 29.28676470588235\n",
            "5 cylinders → 27.366666666666664\n",
            "6 cylinders → 19.985714285714284\n",
            "8 cylinders → 14.963106796116506\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "\n",
        "df = pd.read_csv(\"file.csv\")\n",
        "\n",
        "df.columns = df.columns.str.strip().str.replace(\" \", \"_\")\n",
        "\n",
        "\n",
        "mpg = df[\"mpg\"].to_numpy()\n",
        "\n",
        "mean_mpg = np.mean(mpg)\n",
        "median_mpg = np.median(mpg)\n",
        "std_mpg = np.std(mpg)\n",
        "\n",
        "count_above_25 = np.sum(mpg > 25)\n",
        "\n",
        "print(\"Mean MPG:\", mean_mpg)\n",
        "print(\"Median MPG:\", median_mpg)\n",
        "print(\"Std MPG:\", std_mpg)\n",
        "print(\"Cars with MPG > 25:\", count_above_25)\n",
        "\n",
        "\n",
        "cyl = df[\"cylinders\"].to_numpy()\n",
        "car_names = df[\"car_name\"].to_numpy()\n",
        "\n",
        "filtered_cars = car_names[cyl > 6]\n",
        "\n",
        "print(\"\\nCars with >6 cylinders:\")\n",
        "print(filtered_cars.tolist())\n",
        "\n",
        "hp = pd.to_numeric(df[\"horsepower\"], errors=\"coerce\").to_numpy()\n",
        "\n",
        "hp_mean = np.nanmean(hp)\n",
        "hp_clean = np.where(np.isnan(hp), hp_mean, hp)\n",
        "\n",
        "hp_increased = hp_clean * 1.10\n",
        "\n",
        "print(\"\\nHorsepower increased by 10%:\")\n",
        "print(hp_increased[:10])\n",
        "\n",
        "\n",
        "origin = df[\"origin\"].to_numpy()\n",
        "displacement = df[\"displacement\"].to_numpy()\n",
        "\n",
        "europe_disp = displacement[origin == 2]\n",
        "avg_europe_disp = np.mean(europe_disp)\n",
        "\n",
        "print(\"\\nAverage displacement (Europe):\", avg_europe_disp)\n",
        "\n",
        "\n",
        "unique_cyl = np.unique(cyl)\n",
        "\n",
        "print(\"\\nMean MPG by cylinders:\")\n",
        "for c in unique_cyl:\n",
        "    mean_val = np.mean(mpg[cyl == c])\n",
        "    print(f\"{c} cylinders → {mean_val}\")"
      ]
    }
  ]
}