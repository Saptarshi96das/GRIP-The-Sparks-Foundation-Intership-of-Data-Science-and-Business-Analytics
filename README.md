# GRIP-The-Sparks-Foundation-Intership-of-Data-Science-and-Business-Analytics
Task 1 - Prediction using Supervised ML, In this regressor model we need to predict the percentage of an student based on the no. of study hours. This is a simple linear regression model involving just 2 variables, Hours and Scores
Author : SAPTARSHI DAS
IMPORTING THE LIBRARIES
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# GRIP : The Sparks Foundation\n",
    "\n",
    "# Author : Utkaarsh Bhaskarwar\n",
    "\n",
    "# Task 1 - Prediction using  Supervised ML\n",
    "\n",
    "In this regressor model we need to predict the percentage of an student based on the no. of study hours.\n",
    "This is a simple linear regression model involving just 2 variables, Hours and Scores\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## IMPORTING THE LIBRARIES"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## IMPORTING THE DATASET"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "url=\"https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv\"\n",
    "data = pd.read_csv(url)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## EXPLORING THE DATASET"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2.5</td>\n",
       "      <td>21</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5.1</td>\n",
       "      <td>47</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.2</td>\n",
       "      <td>27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8.5</td>\n",
       "      <td>75</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.5</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Hours  Scores\n",
       "0    2.5      21\n",
       "1    5.1      47\n",
       "2    3.2      27\n",
       "3    8.5      75\n",
       "4    3.5      30"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(25, 2)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 25 entries, 0 to 24\n",
      "Data columns (total 2 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   Hours   25 non-null     float64\n",
      " 1   Scores  25 non-null     int64  \n",
      "dtypes: float64(1), int64(1)\n",
      "memory usage: 528.0 bytes\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## PLOTTING THE DISTRIBUTIONS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "data.plot(x='Hours', y='Scores', style='o')  \n",
    "plt.title('Hours vs Percentage')  \n",
    "plt.xlabel('Hours Studied')  \n",
    "plt.ylabel('Percentage Score')  \n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## PREPARING THE DATA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "x=data.drop('Scores',axis=1)\n",
    "y=data['Scores']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## SPLITTING THE DATASET INTO TRAINING AND TEST DATA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>1.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>7.7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>3.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>8.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>9.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>6.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>2.7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>5.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>2.7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>4.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>8.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>4.5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Hours\n",
       "14    1.1\n",
       "1     5.1\n",
       "10    7.7\n",
       "13    3.3\n",
       "8     8.3\n",
       "6     9.2\n",
       "18    6.1\n",
       "4     3.5\n",
       "9     2.7\n",
       "7     5.5\n",
       "20    2.7\n",
       "3     8.5\n",
       "0     2.5\n",
       "21    4.8\n",
       "15    8.9\n",
       "12    4.5"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(16, 1)"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>1.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>7.4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>2.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>5.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>3.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>1.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>7.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>6.9</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Hours\n",
       "5     1.5\n",
       "2     3.2\n",
       "19    7.4\n",
       "16    2.5\n",
       "11    5.9\n",
       "22    3.8\n",
       "17    1.9\n",
       "24    7.8\n",
       "23    6.9"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(9, 1)"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_test.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## TRAINING THE MODEL ON TRAIN DATA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "le = LinearRegression()\n",
    "le.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([17.04289179, 33.51695377, 74.21757747, 26.73351648, 59.68164043,\n",
       "       39.33132858, 20.91914167, 78.09382734, 69.37226512])"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred=le.predict(x_test)\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Actual</th>\n",
       "      <th>Predicted</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>20</td>\n",
       "      <td>17.042892</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>27</td>\n",
       "      <td>33.516954</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>69</td>\n",
       "      <td>74.217577</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>30</td>\n",
       "      <td>26.733516</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>62</td>\n",
       "      <td>59.681640</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>35</td>\n",
       "      <td>39.331329</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>24</td>\n",
       "      <td>20.919142</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>86</td>\n",
       "      <td>78.093827</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>76</td>\n",
       "      <td>69.372265</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Actual  Predicted\n",
       "5       20  17.042892\n",
       "2       27  33.516954\n",
       "19      69  74.217577\n",
       "16      30  26.733516\n",
       "11      62  59.681640\n",
       "22      35  39.331329\n",
       "17      24  20.919142\n",
       "24      86  78.093827\n",
       "23      76  69.372265"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  \n",
    "df "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## VISUALISING THE TRAINING SET RESULTS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "line = le.coef_*x+le.intercept_\n",
    "\n",
    "plt.scatter(x,y)\n",
    "plt.xlabel(\"Hours\")\n",
    "plt.ylabel(\"Score\")\n",
    "plt.plot(x,line)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## PREDICTING THE TEST RESULTS\n",
    "### ACCURACY OF OUR MODEL"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9501107277744313"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "le.score(x_train,y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## PREDICTING RESULTS FOR 9.25 HOURS OF STUDY"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "No of Hours = 9.25\n",
      "Predicted Score = 92.14523314523314\n"
     ]
    }
   ],
   "source": [
    "hours = 9.25\n",
    "own_pred = le.predict(np.array(hours).reshape(-1,1))\n",
    "print(\"No of Hours = {}\".format(hours))\n",
    "print(\"Predicted Score = {}\".format(own_pred[0]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## EVALUATION OF MODEL"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Absolute Error: 4.691397441397438\n"
     ]
    }
   ],
   "source": [
    "from sklearn import metrics  \n",
    "print('Mean Absolute Error:', \n",
    "      metrics.mean_absolute_error(y_test, y_pred)) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
