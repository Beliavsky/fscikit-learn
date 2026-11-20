program reference_recursive_selection_driver
  use sklearn, only: dp,rfe,rfecv,linear_regression,ridge,elastic_net,lasso,logistic_regression
  implicit none
  type(rfe) :: selector
  type(rfecv) :: search
  type(linear_regression) :: ols
  type(ridge) :: reg
  type(elastic_net) :: elastic
  type(lasso) :: sparse
  type(logistic_regression) :: classifier
  real(dp), allocatable :: x(:,:),y(:)
  integer, allocatable :: binary(:),multi(:)
  integer :: input,output,n,p,i,j
  character(512) :: path
  character(24) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),y(n),binary(n),multi(n))
  do i=1,n
    read(input,*) x(i,:),y(i),binary(i),multi(i)
  end do
  close(input); call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  reg%alpha=1.3_dp; elastic%alpha=.05_dp; elastic%l1_ratio=.4_dp; elastic%tol=1.e-11_dp
  sparse%alpha=.08_dp; sparse%tol=1.e-11_dp; classifier%c=.8_dp; classifier%tol=1.e-11_dp
  do j=1,20
    write(prefix,'(a,i0)') 'case',j
    ols%fit_intercept=j/=19; classifier%fit_intercept=j/=20
    if(j>=11.and.j<=19) then
      search=rfecv(); search%cv=3
      select case(j)
      case(12); search%min_features_to_select=2; search%step=2; search%cv=4
      case(13); search%step=.35_dp
      case(14); search%min_features_to_select=3
      case(16); search%min_features_to_select=2; search%step=2
      case(17); search%min_features_to_select=p+2; search%step=.3_dp
      case(18); search%step=.4_dp; search%cv=4
      case(19); search%step=100
      end select
      call fit_case(search,j); call emit_model(search)
    else
      selector=rfe()
      select case(j)
      case(2); selector%n_features_to_select=1; selector%step=2
      case(3); selector%n_features_to_select=p-1; selector%step=.35_dp
      case(4,7); selector%n_features_to_select=2
      case(5); selector%n_features_to_select=2; selector%step=2
      case(6); selector%n_features_to_select=p+2; selector%step=100
      case(8); selector%n_features_to_select=3; selector%step=.4_dp
      case(9); selector%n_features_to_select=p; selector%step=2
      case(20); selector%n_features_to_select=1; selector%step=.2_dp
      end select
      call fit_case(selector,j); call emit_model(selector)
    end if
  end do
  close(output)
contains
  subroutine fit_case(model,k)
    class(rfe), intent(inout) :: model
    integer, intent(in) :: k
    real(dp), allocatable :: z(:,:)
    select case(k)
    case(3,12); z=model%fit_transform(x,y,reg)
    case(4,13); z=model%fit_transform(x,y,elastic)
    case(5,14); z=model%fit_transform(x,y,sparse)
    case(7,9,15,17,20); z=model%fit_transform(x,binary,classifier)
    case(8,10,16,18); z=model%fit_transform(x,multi,classifier)
    case default; z=model%fit_transform(x,y,ols)
    end select
  end subroutine
  subroutine emit_model(model)
    class(rfe), intent(in) :: model
    call vector('_support',real(merge(1,0,model%get_support()),dp))
    call vector('_ranking',real(model%ranking_,dp))
    call vector('_shape',real([model%n_features_in_,model%n_features_],dp))
    call matrix('_train',model%transform(x)); call matrix('_query',model%transform(x+.137_dp))
    call matrix('_inverse',model%inverse_transform(model%transform(x)))
    if(allocated(model%regression_estimator_)) then
      call vector('_predict',model%predict(x)); call vector('_predict_query',model%predict(x+.137_dp))
      call vector('_score',[model%score(x,y)]); call vector('_coef',model%regression_estimator_%coef_)
      call vector('_intercept',[model%regression_estimator_%intercept_])
    else
      call vector('_predict',real(model%predict_classes(x),dp))
      call vector('_predict_query',real(model%predict_classes(x+.137_dp),dp))
      if(j==8.or.j==10.or.j==16.or.j==18) then
        call vector('_score',[model%score(x,multi)])
      else
        call vector('_score',[model%score(x,binary)])
      end if
      call matrix('_coef',model%classifier_estimator_%coef_)
      call vector('_intercept',model%classifier_estimator_%intercept_)
      call vector('_classes',real(model%classifier_estimator_%classes_,dp))
      call matrix('_proba',model%predict_proba(x+.137_dp)); call matrix('_decision',model%decision_function(x+.137_dp))
    end if
    select type(model)
    type is(rfecv)
      call vector('_cv_counts',real(model%cv_results_%n_features,dp))
      call vector('_cv_mean',model%cv_results_%mean_test_score)
      call vector('_cv_std',model%cv_results_%std_test_score)
      call matrix('_cv_splits',model%cv_results_%split_test_score)
    end select
  end subroutine
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
