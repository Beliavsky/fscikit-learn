program graphical_function
 use sklearn,only:dp,graphical_lasso,graphical_lasso_result
 implicit none
 type(graphical_lasso_result)::fit
 real(dp)::covariance(3,3)
 covariance=reshape([2._dp,.8_dp,.2_dp,.8_dp,1._dp,.1_dp,.2_dp,.1_dp,1.5_dp],[3,3])
 fit=graphical_lasso(covariance,.2_dp,mode='cd',tol=1.e-8_dp)
 print *,'Iterations: ',fit%n_iter
 print *,'Sparse precision: ',fit%precision
 print *,'Final objective and dual gap: ',fit%costs(size(fit%costs,1),:)
end program
