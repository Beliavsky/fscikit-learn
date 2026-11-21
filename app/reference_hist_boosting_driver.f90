program reference_hist_boosting_driver
  use sklearn_common, only: dp
  use sklearn_hist_gradient_boosting
  implicit none
  type(hist_gradient_boosting_regressor) :: reg
  type(hist_gradient_boosting_classifier) :: clf
  real(dp), allocatable :: x(:,:),q(:,:),y(:),w(:),raw(:,:),query(:,:),staged(:,:,:),proba(:,:)
  integer, allocatable :: labels(:),monotonic(:)
  integer :: kind,n,p,m,iterations,leaves,depth,min_leaf,bins,seed,early,patience,weighted,balanced,warm,j,k
  real(dp) :: rate,l2,tol,quantile
  character(24) :: loss
  read(*,*) kind,n,p,m,iterations,leaves,depth,min_leaf,bins,seed,early,patience,weighted,balanced,warm
  read(*,*) loss,rate,l2,tol,quantile
  allocate(x(n,p),q(m,p),y(n),w(n),labels(n),monotonic(p))
  do j=1,n
    read(*,*) x(j,:)
  end do
  do j=1,m
    read(*,*) q(j,:)
  end do
  if(kind==0) then
    read(*,*) y
  else
    read(*,*) labels
  end if
  read(*,*) w;read(*,*) monotonic
  reg%max_iter=iterations;clf%max_iter=iterations
  reg%max_leaf_nodes=leaves;clf%max_leaf_nodes=leaves
  reg%max_depth=depth;clf%max_depth=depth
  reg%min_samples_leaf=min_leaf;clf%min_samples_leaf=min_leaf
  reg%max_bins=bins;clf%max_bins=bins
  reg%random_state=seed;clf%random_state=seed
  reg%early_stopping=early/=0;clf%early_stopping=early/=0
  reg%n_iter_no_change=patience;clf%n_iter_no_change=patience
  reg%learning_rate=rate;clf%learning_rate=rate;reg%l2_regularization=l2;clf%l2_regularization=l2
  reg%tol=tol;clf%tol=tol
  if(any(monotonic/=0)) then
    reg%monotonic_cst=monotonic;clf%monotonic_cst=monotonic
  end if
  reg%loss=loss;clf%loss=loss;reg%quantile=quantile
  if(balanced/=0) clf%class_weight='balanced'
  if(warm>0) then
    reg%max_iter=warm;clf%max_iter=warm;call fit()
    reg%warm_start=.true.;clf%warm_start=.true.;reg%max_iter=iterations;clf%max_iter=iterations
  end if
  call fit()
  if(kind==0) then
    print *,reg%n_iter_,reg%n_trees_per_iteration_,reg%n_features_in_
    print '( *(es25.16e3,1x))',reg%baseline_prediction_
    raw=reg%raw_predict(x);query=reg%raw_predict(q);staged=reg%staged_raw_predict(q)
    print '( *(es25.16e3,1x))',raw
    print '( *(es25.16e3,1x))',query
    print '( *(es25.16e3,1x))',staged
    print '( *(es25.16e3,1x))',reg%predict(q)
    print *
    print '( *(es25.16e3,1x))',reg%train_score_
    print *
    print *,reg%bin_mapper_%n_bins_non_missing_
    do j=1,p
      print '( *(es25.16e3,1x))',reg%bin_mapper_%bin_thresholds_(j)%values
    end do
    do k=1,reg%n_trees_per_iteration_
      print *,[(size(reg%predictors_(j,k)%nodes_),j=1,reg%n_iter_)]
    end do
  else
    print *,clf%n_iter_,clf%n_trees_per_iteration_,clf%n_features_in_
    print '( *(es25.16e3,1x))',clf%baseline_prediction_
    raw=clf%raw_predict(x);query=clf%raw_predict(q);staged=clf%staged_raw_predict(q)
    print '( *(es25.16e3,1x))',raw
    print '( *(es25.16e3,1x))',query
    print '( *(es25.16e3,1x))',staged
    print *,clf%predict(q)
    proba=clf%predict_proba(q);print '( *(es25.16e3,1x))',proba
    print '( *(es25.16e3,1x))',clf%train_score_
    print *,clf%classes_
    print *,clf%bin_mapper_%n_bins_non_missing_
    do j=1,p
      print '( *(es25.16e3,1x))',clf%bin_mapper_%bin_thresholds_(j)%values
    end do
    do k=1,clf%n_trees_per_iteration_
      print *,[(size(clf%predictors_(j,k)%nodes_),j=1,clf%n_iter_)]
    end do
  end if
contains
  subroutine fit()
    if(kind==0) then
      if(weighted/=0) then
        call reg%fit(x,y,w)
      else
        call reg%fit(x,y)
      end if
    else
      if(weighted/=0) then
        call clf%fit(x,labels,w)
      else
        call clf%fit(x,labels)
      end if
    end if
  end subroutine
end program
