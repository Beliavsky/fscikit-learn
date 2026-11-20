program reference_adaboost_regression_driver
  use sklearn, only: dp,adaboost_regressor
  implicit none
  type(adaboost_regressor) :: m
  real(dp), allocatable :: x(:,:),y(:),w(:),s(:,:),v(:)
  integer :: n,p,i,j,k,input,output
  character(512) :: path
  character(24) :: prefix,key
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),y(n),w(n))
  read(input,*) m%n_estimators,m%random_state,m%max_depth,m%min_samples_leaf,m%min_samples_split,m%max_features
  read(input,*) m%loss,m%learning_rate,m%min_impurity_decrease
  do i=1,n
    read(input,*) x(i,:),y(i),w(i)
  end do
  close(input); call m%fit(x,y,w)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  call vector('weights',m%estimator_weights_); call vector('errors',m%estimator_errors_)
  call vector('importance',m%feature_importances_)
  call vector('shape',real([m%n_estimators_,m%n_features_in_],dp))
  call vector('train',m%predict(x)); call vector('query',m%predict(x+.137_dp))
  s=m%staged_predict(x+.137_dp); call vector('staged',reshape(s,[size(s)]))
  call vector('scores',m%staged_score(x,y)); call vector('score',[m%score(x,y)])
  do k=1,m%n_estimators_
    write(prefix,'(a,i0)') 'tree',k
    call vector(trim(prefix)//'_predict',m%estimators_(k)%predict(x+.137_dp))
    call vector(trim(prefix)//'_meta',real([m%estimators_(k)%random_state, &
      m%estimators_(k)%get_depth(),m%estimators_(k)%get_n_leaves()],dp))
    j=m%estimators_(k)%node_count_; allocate(v(j))
    do i=1,j
      v(i)=m%estimators_(k)%nodes_(i)%threshold
    end do
    call vector(trim(prefix)//'_threshold',v)
    do i=1,j
      v(i)=m%estimators_(k)%nodes_(i)%value
    end do
    call vector(trim(prefix)//'_value',v)
    do i=1,j
      v(i)=m%estimators_(k)%nodes_(i)%feature
    end do
    call vector(trim(prefix)//'_feature',v); deallocate(v)
  end do
  close(output)
contains
  subroutine vector(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:)
    write(output,'(a,1x,*(es25.16e3,1x))') name,a
  end subroutine
end program
