program reference_nmf_sparse_driver
  use sklearn_common,only:dp
  use sklearn_sparse,only:csr_matrix
  use sklearn_decomposition,only:nmf,non_negative_factorization
  use sklearn_nmf_mu,only:nmf_sparse_error
  use sklearn_nmf_sparse_data,only:nmf_prepare_sparse
  implicit none
  type(nmf)::model
  type(csr_matrix)::x,q,before,prepared
  real(dp),allocatable::w(:,:),h(:,:),w0(:,:),h0(:,:),z(:,:)
  real(dp)::err
  integer::n,p,count,m,qcount,k,k0,steps,seed,shuffle,route,i,input,output,n_iter
  logical::preserved
  character(1024)::path
  call get_command_argument(1,path);open(newunit=input,file=trim(path),status='old')
  read(input,*)n,p,count,m,qcount,k,k0,steps,seed,shuffle,route
  read(input,*)model%solver,model%init,model%beta_loss,model%tol,model%alpha_w,model%alpha_h,model%l1_ratio
  x%n_rows=n;x%n_cols=p;q%n_rows=m;q%n_cols=p
  allocate(x%indptr(n+1),x%indices(count),x%data(count),q%indptr(m+1),q%indices(qcount),q%data(qcount))
  read(input,*)x%indptr;read(input,*)x%indices;read(input,*)x%data
  read(input,*)q%indptr;read(input,*)q%indices;read(input,*)q%data
  allocate(w0(n,k0),h0(k0,p));read(input,*)w0;read(input,*)h0;close(input)
  model%n_components=k;model%max_iter=steps;model%random_state=seed;model%shuffle=shuffle/=0;before=x
  print '(a)','nmf_stage fit'
  if(route==0)then
    w=model%fit_transform(x,w0,h0);h=model%components_;n_iter=model%n_iter_;err=model%reconstruction_err_
  else if(route==1)then
    call model%fit(x,w0,h0);w=model%transform(x);h=model%components_;n_iter=model%n_iter_;err=model%reconstruction_err_
  else
    call non_negative_factorization(x,w,h,n_iter,n_components=k,w0=w0,h0=h0,init=model%init, &
      update_h=route==2,solver=model%solver,beta_loss=model%beta_loss,tol=model%tol,max_iter=steps, &
      alpha_w=model%alpha_w,alpha_h=model%alpha_h,l1_ratio=model%l1_ratio,random_state=seed,shuffle=shuffle/=0)
    prepared=nmf_prepare_sparse(x,model%beta_loss,model%solver);err=nmf_sparse_error(prepared,w,h,model%beta_loss)
    model%components_=h;model%n_components_=size(h,1);model%n_features_in_=p
  end if
  print '(a)','nmf_stage query'
  z=model%transform(q)
  preserved=all(x%indptr==before%indptr).and.all(x%indices==before%indices).and.all(x%data==before%data)
  call get_command_argument(2,path);open(newunit=output,file=trim(path),status='replace')
  write(output,'(*(es25.16e3,1x))')reshape(w,[size(w)])
  write(output,'(*(es25.16e3,1x))')reshape(h,[size(h)])
  write(output,'(*(es25.16e3,1x))')reshape(z,[size(z)])
  write(output,'(es25.16e3)')err
  write(output,'(*(i0,1x))')n_iter,model%n_components_,model%n_features_in_,merge(1,0,preserved)
  close(output)
end program
