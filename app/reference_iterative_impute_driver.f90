program reference_iterative_impute_driver
  use sklearn_iterative_impute
  use sklearn_common, only: dp
  implicit none
  type(iterative_imputer) :: model
  real(dp), allocatable :: x(:,:),q(:,:),z(:,:),a(:,:),b(:,:)
  character(128), allocatable :: names(:)
  integer :: n,p,m,j,posterior,skip,keep,indicator,sentinel,bounds
  read(*,*) n,p,m,model%max_iter,model%n_nearest_features,model%random_state
  read(*,*) posterior,skip,keep,indicator,sentinel,bounds
  read(*,*) model%initial_strategy,model%imputation_order
  read(*,*) model%tol,model%fill_value,model%min_value,model%max_value
  if(bounds/=0) then
    allocate(model%min_values(p),model%max_values(p)); read(*,*) model%min_values; read(*,*) model%max_values
  end if
  model%sample_posterior=posterior/=0;model%skip_complete=skip/=0;model%keep_empty_features=keep/=0
  model%add_indicator=indicator/=0;model%missing_is_nan=sentinel==0;model%missing_values=-99
  allocate(x(n,p),q(m,p))
  do j=1,n
    read(*,*) x(j,:)
  end do
  do j=1,m
    read(*,*) q(j,:)
  end do
  z=model%fit_transform(x); a=model%transform(q); b=model%transform(q); names=model%get_feature_names_out()
  print *,model%n_iter_,model%n_features_with_missing_,size(model%imputation_sequence_),size(z,2)
  print '( *(es25.16e3,1x))',z
  print '( *(es25.16e3,1x))',a
  print '( *(es25.16e3,1x))',b
  print '( *(es25.16e3,1x))',model%initial_statistics_
  print *,model%valid_features_
  print *,model%indicator_features_
  print '( *(a,1x))',names
  print '( *(es25.16e3,1x))',x
  print '( *(es25.16e3,1x))',q
  do j=1,size(model%imputation_sequence_)
    print *,model%imputation_sequence_(j)%feat_idx,size(model%imputation_sequence_(j)%neighbor_feat_idx)
    print *,model%imputation_sequence_(j)%neighbor_feat_idx
    print '( *(es25.16e3,1x))',model%imputation_sequence_(j)%estimator%alpha_, &
      model%imputation_sequence_(j)%estimator%lambda_,model%imputation_sequence_(j)%estimator%intercept_, &
      model%imputation_sequence_(j)%estimator%coef_
    print '( *(es25.16e3,1x))',model%imputation_sequence_(j)%estimator%sigma_
  end do
end program
