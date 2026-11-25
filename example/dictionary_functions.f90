program dictionary_functions
  use sklearn,only:dp,dict_learning,dict_learning_online,dictionary_learning_result,online_dictionary_result
  implicit none
  type(dictionary_learning_result)::batch
  type(online_dictionary_result)::online
  real(dp)::x(6,2)
  x(:,1)=[1._dp,2._dp,3._dp,0._dp,0._dp,1._dp]
  x(:,2)=[0._dp,0._dp,1._dp,1._dp,2._dp,3._dp]
  batch=dict_learning(x,2,.1_dp,max_iter=20,method='cd',random_state=7)
  online=dict_learning_online(x,2,alpha=.1_dp,max_iter=20,batch_size=3,method='cd',random_state=7)
  print *,'Batch objective:',batch%errors(size(batch%errors))
  print *,'Online steps:',online%n_steps
  print *,'Online reconstruction error:',sum((x-matmul(online%code,online%dictionary))**2)
end program
