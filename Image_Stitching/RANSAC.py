import numpy as np
import scipy
import scipy.linalg

def ransac(data,tranformed_data,model,n,k,t,d,debug=False,return_all=False): 
#n = 5 要取幾個點, k = 5000 iterations, t = 7e4 threshold for inlier selection, d = 50 numbers of inliers
    iterations = 0
    best_fit = 0
    best_err = np.inf
    best_inlier_idxs = None

    while(iterations < k):
        print("iterations:")
        print(iterations)
        maybe_idxs,test_idxs = random_partition(n,data.shape[0])
        # print("maybe_idxs:")
        # print(maybe_idxs)
        # print("test_idxs:")
        # print(test_idxs)
        maybe_inliers = data[maybe_idxs,:]
        # print("data:")
        # print(data)
        # print("maybe_inliers:")
        # print(maybe_inliers)
        test_points = data[test_idxs]
        # print("test_points:")
        # print(test_points)
        maybe_inliers_output = tranformed_data[maybe_idxs,:]
        test_points_output = tranformed_data[test_idxs]
        maybe_model = model.fit(maybe_inliers,maybe_inliers_output)
        vote_idxs = model.get_error_idxs(test_points,test_points_output,maybe_model,t)
        # print("vote_idxs:")
        # print(vote_idxs)
        also_idxs = [idx+1 for idx in vote_idxs ] # transform vote_idxs to also idxs(test_data idx to data idxs)
        # print("also_idxs:")
        # print(also_idxs)
        #count error
        # print("test_points:")
        # print(test_points)
        # print(test_points.shape[0])
        # for i in range(test_points.shape[0]):
        #     test = test_points_output 
        #     A = test_points[i]
        #     print("A")
        #     print(A)
        #     B = test_points_output[i]
        #     test_err = model.get_error(A,B,maybe_model)
        # print("t:")
        # print(t)
        # print("test_err")
        # print(test_err)
        
        # also_idxs = test_idxs[test_err < t] # select indices of rows with accepted points
        
        also_inliers = test_points[vote_idxs,:]
        # print("test_points:")
        # print(test_points)
        # print(also_inliers)
        also_inliers_output = test_points_output[vote_idxs,:]
        # if(debug):
        #     print 'test_err.min()',test_err.min()
        #     print 'test_err.max()',test_err.max()
        #     print 'numpy.mean(test_err)',numpy.mean(test_err)
        #     print 'iteration %d:len(alsoinliers) = %d'%(
        #         iterations,len(alsoinliers))
        # print("d:")
        # print(d)
        if(len(also_inliers)<d):
            better_data = np.concatenate((maybe_inliers,also_inliers))
            better_data_output = np.concatenate((maybe_inliers_output,also_inliers_output))
            better_model = model.fit(better_data,better_data_output)
            better_err = model.get_error(better_data,better_data_output,better_model)
            print("better_err:")
            print(better_err)
            # this_err = np.mean(better_err)
            this_err = better_err
            print("best_err:")
            print(best_err)
            if(this_err < best_err):
                print("enter")
                best_fit = better_model
                best_err = this_err
                best_inlier_idxs = np.concatenate((maybe_idxs,also_idxs))
        iterations += 1
    if(best_fit is None):
        raise ValueError("did not meet fit acceptance criteria")
    if(return_all):
        return best_fit, {'inliers':best_inlier_idxs}
    else:
        return best_fit


def random_partition(n,n_data):
     """return n random rows of data (and also the other len(data)-n rows)"""
     all_idxs = np.arange(n_data)
     # print(all_idxs)
     np.random.shuffle(all_idxs)
     # print(all_idxs)
     idxs1 = all_idxs[:n]
     # print(n)
     # print("idxs1")
     # print(idxs1)
     idxs2 = all_idxs[n:]
     # print("idxs2")
     # print(idxs2)
     return idxs1,idxs2    


class LinearLeastSquaresModel:
    """
    linear system solved using linear least squares
    This class serves as an example that fulfills the model interface
    needed by the ransac() function.
    
    """  
    def __init__(self,input_columns,output_columns,debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug
    def fit(self,data,transformed_data):
        # A = np.vstack([data[i,:] for i in range(self.input_columns)]).T
        # B = np.vstack([transformed_data[i,:] for i in range(self.output_columns)]).T

        # A = data.T
        # B = transformed_data.T

        #ax+by = c
        # x1 = data[0]
        # y1 = data[1]
        # x2 = transformed_data[0]
        # y2 = transformed_data[1]
        # a = (y2-y1)/(x2-x1)
        # b = y1 - a * x1

        #要算translation matrix, M * A = B 
        # print(data)
        A = data.T
        # print(A)
        B = transformed_data.T
        x_translate = B[0][0] - A[0][0]
        y_translate = B[1][0] - A[1][0]
        M = np.array([[1,0,x_translate],
                      [0,1,y_translate]])

        # print(M)
        # x,resids,rank,s = np.linalg.lstsq(A,B)
        # print(x)
        return M
    def get_error_idxs(self,data,transformed_data,model,threshold):
        # A = np.vstack([data[i,:] for i in range(self.input_columns)]).T
        # B = np.vstack([transformed_data[i,:] for i in range(self.output_columns)]).T 
        # print(data)
        A = data.T
        # print(data)
        # print(A.shape)
        # print(A)
        B = transformed_data.T
        # print(B.shape)
        vote_idxs = [] #store index which tranformed point is also active 

        for i in range(A.shape[1]):
            # print("i:")
            # print(i)
            x_test = A[0][i]
            y_test = A[1][i]
            # print("x_test:")
            # print(x_test)
            # print("y_test")
            # print(y_test)
            matrix = np.array([[x_test,y_test,1]]).T
            # print("matrix:")
            # print(matrix)
            # print(matrix.shape)
            # print("model:")
            # print(model)
            # print(model.shape)
            matrix_predict = scipy.dot(model,matrix)
            # print("matrix_predict:")
            # print(matrix_predict.shape)
            # print(matrix_predict)  

            x_correct = B[0][i]
            y_correct = B[1][i]
            matrix_correct = np.array([[x_correct,y_correct]]).T
            # print("matrix_correct:")
            # print(matrix_correct)

            err_per_point = np.sum((matrix_correct - matrix_predict)**2) 
            # print("err_per_point:")
            # print(err_per_point)
            # print("threshold:")
            # print(threshold)
            if(err_per_point < threshold):
                # print("store")
                vote_idxs.append(i)
                # print(vote_idxs)
        return vote_idxs

    def get_error(self,data,transformed_data,model):
        # print("get_error")
        A = data.T
        # print(data)
        # print(A.shape)
        # print(A)
        B = transformed_data.T
        # print(B.shape)
        error = [] #count error 

        for i in range(A.shape[1]):
            # print("i:")
            # print(i)
            x_test = A[0][i]
            y_test = A[1][i]
            # print("x_test:")
            # print(x_test)
            # print("y_test")
            # print(y_test)
            matrix = np.array([[x_test,y_test,1]]).T
            # print("matrix:")
            # print(matrix)
            # print(matrix.shape)
            # print("model:")
            # print(model)
            # print(model.shape)
            matrix_predict = scipy.dot(model,matrix)
            # print("matrix_predict:")
            # print(matrix_predict.shape)
            # print(matrix_predict)  

            x_correct = B[0][i]
            y_correct = B[1][i]
            matrix_correct = np.array([[x_correct,y_correct]]).T
            # print("matrix_correct:")
            # print(matrix_correct)

            error.append(np.sum((matrix_correct - matrix_predict)**2))
            # print("error:")
            # print(error)
            # print("threshold:")
            # print(threshold)
        error_sum = np.sum(error)
        # print("error_sum:")
        # print(error_sum)
        return error_sum

def test():
    # # generate perfect input data
    # n_samples = 500
    # n_inputs = 1
    # n_outputs = 1
    # A_exact = 20*np.random.random((n_samples,n_inputs))
    # perfect_fit = 60*np.random.normal(size=(n_inputs,n_outputs)) # the model
    # B_exact = scipy.dot(A_exact,perfect_fit)
    # assert B_exact.shape == (n_samples,n_outputs)

    # # add a little gaussian noise (linear least squares alone should handle this well)
    # A_noisy = A_exact + np.random.normal(size=A_exact.shape )
    # B_noisy = B_exact + np.random.normal(size=B_exact.shape )

    # if 1:
    #     # add some outliers
    #     n_outliers = 100
    #     all_idxs = np.arange( A_noisy.shape[0] )
    #     np.random.shuffle(all_idxs)
    #     outlier_idxs = all_idxs[:n_outliers]
    #     non_outlier_idxs = all_idxs[n_outliers:]
    #     A_noisy[outlier_idxs] =  20*np.random.random((n_outliers,n_inputs) )
    #     B_noisy[outlier_idxs] = 50*np.random.normal(size=(n_outliers,n_outputs) )

    # setup model
    # all_data = np.hstack( (A_noisy,B_noisy) )


    # input_columns = range(n_inputs) # the first columns of the array
    # output_columns = [n_inputs+i for i in range(n_outputs)] # the last columns of the array
    
    #matrix1儲存原本feature pos, matrix2儲存transformed後的feature pos
    matrix1 = np.array(([1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8]))
    matrix2 = np.array(([6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]))
    # print(matrix1)
    # print(matrix1.shape)

    input_columns = matrix1.shape[0] # 原本位置
    output_columns = matrix2.shape[0] # 轉換後位置

    debug = True
    model = LinearLeastSquaresModel(input_columns,output_columns,debug=debug)
    
    # linear_fit,resids,rank,s = np.linalg.lstsq(all_data[:,input_columns],all_data[:,output_columns])

    # run RANSAC algorithm
    # ransac_fit, ransac_data = ransac(all_data,model,
    #                                  5, 5000, 7e4, 50, # misc. parameters
    #                                  debug=debug,return_all=True)
    ransac_fit, ransac_data = ransac(matrix1,matrix2,model,
                                     1, 5000, 5, 50, # misc. parameters
                                     debug=debug,return_all=True)

    print("ransac_fit:")
    print(ransac_fit)
    print("ransac_data:")
    print(ransac_data)
    # if 1:
    #     import pylab

    #     sort_idxs = np.argsort(A_exact[:,0])
    #     A_col0_sorted = A_exact[sort_idxs] # maintain as rank-2 array

    #     if 1:
    #         pylab.plot( A_noisy[:,0], B_noisy[:,0], 'k.', label='data' )
    #         pylab.plot( A_noisy[ransac_data['inliers'],0], B_noisy[ransac_data['inliers'],0], 'bx', label='RANSAC data' )
    #     else:
    #         pylab.plot( A_noisy[non_outlier_idxs,0], B_noisy[non_outlier_idxs,0], 'k.', label='noisy data' )
    #         pylab.plot( A_noisy[outlier_idxs,0], B_noisy[outlier_idxs,0], 'r.', label='outlier data' )
    #     pylab.plot( A_col0_sorted[:,0],
    #                 np.dot(A_col0_sorted,ransac_fit)[:,0],
    #                 label='RANSAC fit' )
    #     pylab.plot( A_col0_sorted[:,0],
    #                 np.dot(A_col0_sorted,perfect_fit)[:,0],
    #                 label='exact system' )
    #     pylab.plot( A_col0_sorted[:,0],
    #                 np.dot(A_col0_sorted,linear_fit)[:,0],
    #                 label='linear fit' )
    #     pylab.legend()
    #     pylab.show()

if __name__=='__main__':
    test()

