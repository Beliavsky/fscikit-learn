program reference_isolation_driver
  use sklearn_isolation_forest, only: isolation_forest
  use sklearn_common, only: dp
  implicit none
  type(isolation_forest) :: model
  real(dp), allocatable :: x(:,:),query(:,:),w(:),values(:),z(:,:)
  integer :: n,p,nq,boot,warm,input,output,i,k,j,old
  character(512) :: path
  character(32) :: prefix,key
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p,nq,boot,warm
  read(input,*) model%n_estimators,model%max_samples,model%max_features,model%random_state
  read(input,*) model%max_samples_fraction,model%max_features_fraction,model%contamination
  allocate(x(n,p),w(n),query(nq,p))
  do i=1,n
    read(input,*) x(i,:),w(i)
  end do
  do i=1,nq
    read(input,*) query(i,:)
  end do
  close(input); model%bootstrap=boot==1
  if(warm==1) then
    old=model%n_estimators; model%n_estimators=3; call model%fit(x,w)
    model%warm_start=.true.; model%n_estimators=old
  end if
  call model%fit(x,w)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  call vector('scores',model%score_samples(query)); call vector('decision',model%decision_function(query))
  call vector('predict',real(model%predict(query),dp)); call vector('train',model%score_samples(x))
  call vector('meta',[model%offset_,real(model%n_estimators_,dp),real(model%max_samples_,dp), &
    real(model%max_features_,dp),real(model%n_features_in_,dp)])
  do k=1,model%n_estimators_
    write(prefix,'(a,i0)') 'tree',k
    call vector(trim(prefix)//'_features',real(model%estimators_features_(k)%values,dp))
    call vector(trim(prefix)//'_samples',real(model%estimators_samples_(k)%values,dp))
    call vector(trim(prefix)//'_meta',real([model%estimators_(k)%random_state,model%estimators_(k)%get_depth(), &
      model%estimators_(k)%get_n_leaves()],dp))
    j=model%estimators_(k)%node_count_; allocate(values(j))
    do i=1,j
      values(i)=model%estimators_(k)%nodes_(i)%threshold
    end do
    call vector(trim(prefix)//'_threshold',values)
    do i=1,j
      values(i)=model%estimators_(k)%nodes_(i)%feature
    end do
    call vector(trim(prefix)//'_feature',values)
    do i=1,j
      values(i)=model%estimators_(k)%nodes_(i)%n_samples
    end do
    call vector(trim(prefix)//'_count',values)
    do i=1,j
      values(i)=model%estimators_(k)%nodes_(i)%value
    end do
    call vector(trim(prefix)//'_value',values)
    do i=1,j
      values(i)=model%estimators_(k)%nodes_(i)%impurity
    end do
    call vector(trim(prefix)//'_impurity',values); deallocate(values)
    call vector(trim(prefix)//'_importance',model%estimators_(k)%feature_importances_)
  end do
  close(output)
contains
  subroutine vector(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:)
    write(output,'(a,1x,*(es25.16e3,1x))') name,a
  end subroutine
end program
