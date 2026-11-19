program reference_gradient_boosting_driver
  use sklearn
  implicit none
  type(gradient_boosting_classifier) :: model
  real(dp), allocatable :: x(:,:),query(:,:),z(:,:),cube(:,:,:)
  integer, allocatable :: binary(:),multi(:),target(:),leaves(:,:,:),predictions(:,:)
  integer :: n,p,i,j,t,k,input,output
  character(1024) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),binary(n),multi(n))
  do i=1,n
    read(input,*) x(i,:),binary(i),multi(i)
  end do
  close(input); query=x+.137_dp
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,10
    model=gradient_boosting_classifier(); model%n_estimators=8; model%max_depth=1; target=binary
    select case(j)
    case(2)
      model%max_depth=2; model%n_estimators=9; model%learning_rate=.2_dp; model%init_zero=.true.
    case(3)
      model%max_depth=3; model%n_estimators=6; model%learning_rate=.3_dp
      model%min_samples_leaf=3; model%min_samples_split=6
    case(4); model%min_impurity_decrease=1; model%n_estimators=4
    case(5); target=multi
    case(6)
      target=multi; model%max_depth=2; model%learning_rate=.15_dp; model%init_zero=.true.
    case(7)
      target=multi; model%max_depth=3; model%n_estimators=6; model%learning_rate=.2_dp; model%min_samples_leaf=3
    case(8); target=multi; model%min_impurity_decrease=1; model%n_estimators=4
    case(9); model%loss='exponential'
    case(10)
      model%loss='exponential'; model%max_depth=2; model%n_estimators=6
      model%learning_rate=.2_dp; model%init_zero=.true.
    end select
    call model%fit(x,target); write(prefix,'(a,i0)') 'boost',j
    call vector('_classes',real(model%classes_,dp)); call vector('_prior',model%class_prior_)
    call vector('_init',model%init_); call vector('_loss',model%train_score_)
    call vector('_importance',model%feature_importances_)
    call vector('_metadata',real([model%n_estimators_,model%n_trees_per_iteration_,model%n_features_in_],dp))
    z=model%decision_function(x); call matrix('_train',z)
    z=model%decision_function(query); call matrix('_query',z)
    z=model%predict_proba(query); call matrix('_proba',z)
    z=model%predict_log_proba(query); call matrix('_logproba',z)
    call vector('_predict',real(model%predict(query),dp)); call vector('_score',[model%score(x,target)])
    cube=model%staged_decision_function(query); call vector('_staged_raw',reshape(cube,[size(cube)]))
    cube=model%staged_predict_proba(query); call vector('_staged_proba',reshape(cube,[size(cube)]))
    predictions=model%staged_predict(query); call matrix('_staged_predict',real(predictions,dp))
    leaves=model%apply(query); call vector('_leaves',real(reshape(leaves,[size(leaves)]),dp))
    if(allocated(cube)) deallocate(cube)
    allocate(cube(n,model%n_estimators_,model%n_trees_per_iteration_))
    do k=1,model%n_trees_per_iteration_
      do t=1,model%n_estimators_
        cube(:,t,k)=model%estimators_(t,k)%predict(query)
      end do
    end do
    call vector('_tree_values',reshape(cube,[size(cube)]))
  end do
  close(output)
contains
  subroutine vector(suffix,values)
    character(*), intent(in) :: suffix
    real(dp), intent(in) :: values(:)
    write(output,'(a,*(1x,es25.16e3))') trim(prefix)//suffix,values
  end subroutine
  subroutine matrix(suffix,values)
    character(*), intent(in) :: suffix
    real(dp), intent(in) :: values(:,:)
    call vector(suffix,reshape(values,[size(values)]))
  end subroutine
end program
