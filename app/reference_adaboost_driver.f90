program reference_adaboost_driver
  use sklearn, only: dp,adaboost_classifier
  implicit none
  type(adaboost_classifier) :: model
  real(dp), allocatable :: x(:,:),w(:),train(:,:),weights(:),cube(:,:,:)
  integer, allocatable :: binary(:),multi(:),y(:),staged(:,:)
  integer :: input,output,n,p,i,j
  character(512) :: path
  character(24) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),w(n),binary(n),multi(n))
  do i=1,n
    read(input,*) x(i,:),w(i),binary(i),multi(i)
  end do
  close(input); call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,6
    model=adaboost_classifier(); model%n_estimators=8; model%random_state=7; train=x; weights=w; y=binary
    select case(j)
    case(2); y=multi
    case(3); train(:,2:)=0; weights=1
    case(4); model%learning_rate=.3_dp; model%n_estimators=12; y=multi
    case(5); model%n_estimators=1
    case(6); train(:,2:)=0; y=merge(7,-4,train(:,1)>0)
    end select
    call model%fit(train,y,sample_weight=weights); write(prefix,'(a,i0)') 'case',j
    call vector('_weights',model%estimator_weights_); call vector('_errors',model%estimator_errors_)
    call vector('_classes',real(model%classes_,dp)); call vector('_importance',model%feature_importances_)
    call matrix('_train',model%decision_function(train)); call matrix('_query',model%decision_function(train+.137_dp))
    call matrix('_proba',model%predict_proba(train+.137_dp)); call vector('_predict',real(model%predict(train+.137_dp),dp))
    cube=model%staged_decision_function(train+.137_dp); call vector('_staged',reshape(cube,[size(cube)]))
    cube=model%staged_predict_proba(train+.137_dp); call vector('_staged_proba',reshape(cube,[size(cube)]))
    staged=model%staged_predict(train+.137_dp); call matrix('_staged_predict',real(staged,dp))
    call vector('_shape',real([model%n_estimators_,model%n_classes_,model%n_features_in_],dp))
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
