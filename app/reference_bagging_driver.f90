program reference_bagging_driver
  use sklearn
  use sklearn_base_dispatch, only: labels,regression,probabilities,base_proba,base_classes
  implicit none
  type(bagging_classifier) :: clf
  type(bagging_regressor) :: reg
  type(decision_tree_classifier) :: tc
  type(decision_tree_regressor) :: tr
  type(extra_tree_classifier) :: ec
  type(extra_tree_regressor) :: er
  type(kneighbors_classifier) :: kc
  type(kneighbors_regressor) :: kr
  type(gaussian_nb) :: nb
  type(linear_regression) :: ols
  type(logistic_regression) :: logistic
  type(ridge) :: rr
  type(ridge_classifier) :: rc
  type(elastic_net) :: en
  class(*), allocatable :: base
  real(dp), allocatable :: x(:,:),q(:,:),y(:),w(:),response(:,:),training(:,:)
  integer :: n,p,m,mode,code,bcount,warmcount,oob,boot,bootf,ms,mf,weighted,seed,i,b,unit,out,stage,stages
  real(dp) :: sf,ff
  character(1024) :: path
  call get_command_argument(1,path); open(newunit=unit,file=trim(path),status='old')
  read(unit,*) n,p,m,mode,code,bcount,warmcount,oob,boot,bootf,ms,mf,weighted,seed
  read(unit,*) sf,ff
  allocate(x(n,p),q(m,p),y(n),w(n))
  do i=1,n
    read(unit,*) x(i,:)
  end do
  read(unit,*) y; read(unit,*) w
  do i=1,m
    read(unit,*) q(i,:)
  end do
  close(unit)
  tc%max_depth=3; tr%max_depth=3; ec%max_depth=3; er%max_depth=3
  kc%n_neighbors=3; kr%n_neighbors=3
  if(ms>0.and.ms<=2) then
    kc%n_neighbors=1; kr%n_neighbors=1
  end if
  logistic%tol=1.e-10_dp; logistic%max_iter=500; logistic%c=.7_dp
  rr%alpha=.7_dp; rc%alpha=.7_dp
  en%alpha=.05_dp; en%l1_ratio=.3_dp; en%tol=1.e-10_dp; en%max_iter=10000
  if(mode==0) then
    select case(code)
    case(0)
      allocate(base,source=tc)
    case(1)
      allocate(base,source=ec)
    case(2)
      allocate(base,source=kc)
    case(3)
      allocate(base,source=nb)
    case(4)
      allocate(base,source=logistic)
    case(5)
      allocate(base,source=rc)
    case(6)
      ec%max_depth=100; allocate(base,source=ec)
    end select
  else
    select case(code)
    case(0)
      allocate(base,source=tr)
    case(1)
      allocate(base,source=er)
    case(2)
      allocate(base,source=kr)
    case(3)
      allocate(base,source=ols)
    case(4)
      allocate(base,source=rr)
    case(5)
      allocate(base,source=en)
    case(6)
      er%max_depth=100; allocate(base,source=er)
    end select
  end if
  clf%n_estimators=bcount; clf%random_state=seed; clf%oob_score=oob==1
  clf%bootstrap=boot==1; clf%bootstrap_features=bootf==1
  clf%max_samples=ms; clf%max_features=mf; clf%max_samples_fraction=sf; clf%max_features_fraction=ff
  reg%n_estimators=bcount; reg%random_state=seed; reg%oob_score=oob==1
  reg%bootstrap=boot==1; reg%bootstrap_features=bootf==1
  reg%max_samples=ms; reg%max_features=mf; reg%max_samples_fraction=sf; reg%max_features_fraction=ff
  call get_command_argument(2,path); open(newunit=out,file=trim(path),status='replace')
  stages=1; if(warmcount>0) stages=2
  do stage=1,stages
    if(stage==2) then
      clf%n_estimators=warmcount; clf%warm_start=.true.
      reg%n_estimators=warmcount; reg%warm_start=.true.
    end if
    if(mode==0) then
      if(weighted==1) then
        call clf%fit(x,int(y),base,w)
      else
        call clf%fit(x,int(y),base)
      end if
      write(out,*) clf%n_estimators_,clf%n_features_in_,clf%n_classes_,clf%max_samples_,clf%max_features_
      write(out,*) clf%classes_; write(out,*) clf%predict(q)
      response=clf%predict_proba(q); training=clf%predict_proba(x)
      call matrix(response); call matrix(training)
      write(out,*) clf%score(x,int(y),w)
      if(oob==1) then
        write(out,*) clf%oob_score_; call matrix(clf%oob_decision_function_)
      end if
      do b=1,clf%n_estimators_
        call member(clf%estimators_(b)%estimator,clf%seeds_(b),clf%estimators_features_(b)%values, &
          clf%estimators_samples_(b)%values)
      end do
    else
      if(weighted==1) then
        call reg%fit(x,y,base,w)
      else
        call reg%fit(x,y,base)
      end if
      write(out,*) reg%n_estimators_,reg%n_features_in_,0,reg%max_samples_,reg%max_features_
      write(out,*) reg%predict(q); write(out,*) reg%predict(x); write(out,*) reg%score(x,y,w)
      if(oob==1) write(out,*) reg%oob_score_,reg%oob_prediction_
      do b=1,reg%n_estimators_
        call member(reg%estimators_(b)%estimator,reg%seeds_(b),reg%estimators_features_(b)%values, &
          reg%estimators_samples_(b)%values)
      end do
    end if
  end do
  call matrix(x); call matrix(q); write(out,*) y; write(out,*) w
  close(out)
contains
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    integer :: row
    do row=1,size(a,1)
      write(out,*) a(row,:)
    end do
  end subroutine
  subroutine member(model,random_seed,features,samples)
    class(*), intent(in) :: model
    integer, intent(in) :: random_seed,features(:),samples(:)
    real(dp), allocatable :: a(:,:)
    integer :: j
    write(out,*) random_seed,size(features),size(samples)
    write(out,*) features; write(out,*) samples
    if(mode==0) then
      if(base_proba(model)) then
        write(out,*) 1; write(out,*) base_classes(model)
        a=probabilities(model,q(:,features)); call matrix(a)
        a=probabilities(model,x(:,features)); call matrix(a)
      else
        write(out,*) 0; write(out,*) labels(model,q(:,features)); write(out,*) labels(model,x(:,features))
      end if
    else
      write(out,*) regression(model,q(:,features)); write(out,*) regression(model,x(:,features))
    end if
    select type(model)
    class is(decision_tree_classifier)
      write(out,*) model%node_count_,model%random_state
      write(out,*) model%feature_importances_
      do j=1,model%node_count_
        associate(a=>model%nodes_(j))
          write(out,*) a%left,a%right,a%feature,a%n_samples,a%depth,merge(1,0,a%missing_go_left)
          write(out,*) a%threshold,a%weighted_n_samples,a%impurity
          write(out,*) a%probability
        end associate
      end do
    class is(decision_tree_regressor)
      write(out,*) model%node_count_,model%random_state
      write(out,*) model%feature_importances_
      do j=1,model%node_count_
        associate(a=>model%nodes_(j))
          write(out,*) a%left,a%right,a%feature,a%n_samples,a%depth,merge(1,0,a%missing_go_left)
          write(out,*) a%threshold,a%weighted_n_samples,a%impurity
          write(out,*) a%value
        end associate
      end do
    type is(linear_regression)
      write(out,*) -1,model%rank_
      write(out,*) model%coef_; write(out,*) model%intercept_
    type is(ridge)
      write(out,*) -2,0
      write(out,*) model%coef_; write(out,*) model%intercept_
    class default
      write(out,*) 0,0
    end select
  end subroutine
end program
