program reference_categorical_smoothing_driver
  use sklearn_categorical_nb, only: categorical_nb
  use sklearn_common, only: dp
  implicit none
  type(categorical_nb) :: model
  real(dp), allocatable :: x(:,:),original(:,:),q(:,:),query(:,:),w(:)
  integer, allocatable :: y(:),classes(:),ends(:),minima(:)
  integer :: n,p,m,k,batches,stream,weighted,fixed,fitprior,force,i,j,start,last
  read(*,*) n,p,m,k,batches,stream,weighted,fixed,fitprior,force,model%alpha,model%min_categories
  model%fit_prior=fitprior/=0; model%force_alpha=force/=0
  allocate(x(n,p),q(m,p),w(n),y(n),classes(k),ends(batches),minima(p))
  read(*,*) x; read(*,*) y; read(*,*) q; read(*,*) w; read(*,*) classes; read(*,*) ends; read(*,*) minima
  if(any(minima>0)) model%min_categories_per_feature=minima
  if(fixed/=0) then
    allocate(model%class_prior(k)); read(*,*) model%class_prior
  end if
  original=x; start=1
  do i=1,batches
    last=ends(i)
    if(stream/=0) then
      if(i==1.or.mod(i,2)==0) then
        if(weighted/=0) then
          call model%partial_fit(x(start:last,:),y(start:last),classes,w(start:last))
        else
          call model%partial_fit(x(start:last,:),y(start:last),classes)
        end if
      else
        if(weighted/=0) then
          call model%partial_fit(x(start:last,:),y(start:last),sample_weight=w(start:last))
        else
          call model%partial_fit(x(start:last,:),y(start:last))
        end if
      end if
    else
      if(weighted/=0) then
        call model%fit(x,y,w)
      else
        call model%fit(x,y)
      end if
    end if
    query=q
    do j=1,p
      query(:,j)=modulo(query(:,j),real(size(model%category_count_(j)%values,2),dp))
    end do
    call vector(real(model%classes_,dp)); call vector(model%class_count_)
    call vector(model%class_log_prior_); call vector(real(model%n_categories_,dp))
    call vector([real(model%n_features_in_,dp)])
    do j=1,p
      call matrix(model%category_count_(j)%values); call matrix(model%feature_log_prob_(j)%values)
    end do
    call matrix(model%joint_log_likelihood(query)); call matrix(model%predict_log_proba(query))
    call matrix(model%predict_proba(query)); call vector(real(model%predict(query),dp))
    call vector(real(model%predict(x(:last,:)),dp)); call vector([model%score(x(:last,:),y(:last))])
    start=last+1
  end do
  if(any(original/=x)) error stop 'CategoricalNB modified original input'
contains
  subroutine vector(value)
    real(dp), intent(in) :: value(:)
    call matrix(reshape(value,[size(value),1]))
  end subroutine
  subroutine matrix(value)
    real(dp), intent(in) :: value(:,:)
    print *, shape(value)
    print '( *(es25.16e3,1x) )',value
  end subroutine
end program
